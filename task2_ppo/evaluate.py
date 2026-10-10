from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import (
    batch_generate,
    response_token_logprobs,
    score_reward_pairs,
)
from common.metrics import sampled_kl, sample_entropy
from common.models import (
    clear_gpu,
    load_policy,
    load_reward_model,
    load_tokenizer,
    reference_mode,
)


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward_model": load_reward_model(cfg)[0],
        "reward_tokenizer": load_reward_model(cfg)[1],
    }


def evaluate_ppo(
    config_path: str,
    adapter_path: str,
    name: str = "standard",
    max_prompts: int | None = 32,
    batch_size: int = 4,
):
    cfg = load_yaml(config_path)
    results_dir = repo_path(cfg.get("results_dir", "results/task2_ppo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(cfg["paths"]["rl_prompt_eval"])
    if max_prompts is not None and max_prompts > 0:
        rows = rows[:max_prompts]

    print(f"[{name}] Starting PPO evaluation:")
    print(f"  Adapter: {adapter_path}")
    print(f"  Prompts: {len(rows)} from {cfg['paths']['rl_prompt_eval']}")

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=adapter_path, trainable=False)
    reward_model, reward_tokenizer = load_reward_model(cfg)

    max_prompt_len = int(cfg.get("max_prompt_length", 256))
    max_resp_len = int(cfg.get("eval_max_response_length", 768))

    prompts = [prompt_messages(r) for r in rows]

    all_rewards = []
    all_kls = []
    all_entropies = []
    all_lengths = []
    all_eos = []
    all_responses = []
    records = []

    for start in tqdm(range(0, len(prompts), batch_size), desc=f"Evaluating [{name}]"):
        chunk_prompts = prompts[start : start + batch_size]
        chunk_rows = rows[start : start + batch_size]

        gen = batch_generate(
            policy,
            tokenizer,
            chunk_prompts,
            max_prompt_length=max_prompt_len,
            max_new_tokens=max_resp_len,
            temperature=float(cfg.get("generation", {}).get("temperature", 0.7)),
            top_p=float(cfg.get("generation", {}).get("top_p", 0.9)),
            do_sample=True,
        )

        with torch.no_grad():
            pol_tok_logp, _ = response_token_logprobs(
                policy,
                gen["sequences"],
                gen["attention_mask"],
                gen["prompt_width"],
                gen["response_ids"],
            )

            with reference_mode(policy):
                ref_tok_logp, _ = response_token_logprobs(
                    policy,
                    gen["sequences"],
                    gen["attention_mask"],
                    gen["prompt_width"],
                    gen["response_ids"],
                )

            rewards = score_reward_pairs(
                reward_model,
                reward_tokenizer,
                chunk_prompts,
                gen["responses"],
                max_length=int(cfg.get("reward_max_length", 1280)),
            ).cpu().tolist()

        for b in range(len(chunk_prompts)):
            mask = gen["response_mask"][b]
            if mask.sum() > 0:
                p_lp = pol_tok_logp[b]
                r_lp = ref_tok_logp[b]
                kl_v = sampled_kl(p_lp, r_lp, mask).item()
                ent_v = sample_entropy(p_lp, mask).item()
            else:
                kl_v = 0.0
                ent_v = 0.0

            all_kls.append(kl_v)
            all_entropies.append(ent_v)

        all_rewards.extend(rewards)
        all_lengths.extend(gen["response_lengths"])
        all_eos.extend(gen["terminated_with_eos"])
        all_responses.extend(gen["responses"])

        for r_row, p_msgs, text, n_tok, r_score, kl_val, eos in zip(
            chunk_rows,
            chunk_prompts,
            gen["responses"],
            gen["response_lengths"],
            rewards,
            all_kls[-len(chunk_prompts):],
            gen["terminated_with_eos"],
        ):
            records.append({
                "source_index": r_row.get("source_index", r_row.get("prompt_id", "unknown")),
                "prompt": p_msgs,
                "response": text,
                "response_tokens": int(n_tok),
                "reward_score": round(float(r_score), 4),
                "kl": round(float(kl_val), 4),
                "has_eos": bool(eos),
            })

    lengths_arr = np.array(all_lengths)
    rewards_arr = np.array(all_rewards)
    kl_arr = np.array(all_kls)
    ent_arr = np.array(all_entropies)

    q25, q75 = np.percentile(lengths_arr, [25, 75]) if len(lengths_arr) else (0, 0)
    iqr = q75 - q25

    metrics = {
        "run_name": name,
        "adapter_path": adapter_path,
        "num_prompts": len(prompts),
        "reward_mean": round(float(np.mean(rewards_arr)), 4),
        "reward_std": round(float(np.std(rewards_arr)), 4),
        "kl_mean": round(float(np.mean(kl_arr)), 4),
        "kl_std": round(float(np.std(kl_arr)), 4),
        "entropy_mean": round(float(np.mean(ent_arr)), 4),
        "length_mean": round(float(np.mean(lengths_arr)), 2),
        "length_std": round(float(np.std(lengths_arr)), 2),
        "length_median": round(float(np.median(lengths_arr)), 2),
        "length_iqr": round(float(iqr), 2),
        "eos_rate": round(float(np.mean(all_eos)), 4),
    }

    out_json = results_dir / f"{name}_eval.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    gen_jsonl = results_dir / f"{name}_generations.jsonl"
    write_jsonl(gen_jsonl, records)

    print("\n" + "=" * 65)
    print(f"PPO Held-Out Evaluation Summary: [{name}]")
    print(f"  Prompts:          {metrics['num_prompts']}")
    print(f"  Reward Score:     {metrics['reward_mean']:.4f} ± {metrics['reward_std']:.4f}")
    print(f"  KL from Reference:{metrics['kl_mean']:.4f} ± {metrics['kl_std']:.4f}")
    print(f"  Policy Entropy:   {metrics['entropy_mean']:.4f}")
    print(f"  Response Length:  {metrics['length_mean']} ± {metrics['length_std']} tokens (IQR: {metrics['length_iqr']})")
    print(f"  EOS Rate:         {metrics['eos_rate'] * 100:.1f}%")
    print("=" * 65)
    print(f"Metrics saved to:     {out_json}")
    print(f"Generations saved to: {gen_jsonl}\n")

    clear_gpu(policy, reward_model)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--max-prompts", type=int, default=32)
    args = ap.parse_args()
    evaluate_ppo(
        config_path=args.config,
        adapter_path=args.adapter,
        name=args.name,
        max_prompts=args.max_prompts,
    )


if __name__ == "__main__":
    main()
