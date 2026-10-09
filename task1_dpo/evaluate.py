from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
    write_jsonl,
)
from common.generation import (
    batch_generate,
    response_sequence_logprobs,
    response_token_logprobs,
    score_reward_pairs,
)
from common.metrics import masked_mean, mean_response_length, sampled_kl, sample_entropy
from common.models import (
    clear_gpu,
    load_policy,
    load_reward_model,
    load_tokenizer,
    reference_mode,
)
from task1_dpo.dpo import dpo_loss


def make_eval_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        valid_indices = []
        for idx, row in enumerate(rows):
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            try:
                c_ids, c_mask = encode_prompt_response(tokenizer, prompt, yc, max_length)
                r_ids, r_mask = encode_prompt_response(tokenizer, prompt, yr, max_length)
                chosen.append((c_ids, c_mask))
                rejected.append((r_ids, r_mask))
                valid_indices.append(idx)
            except ValueError:
                continue
        if not chosen:
            return None, None, []
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected), valid_indices
    return collate


def evaluate_dpo(
    config_path: str,
    adapter_path: str,
    name: str = "standard",
    eval_dataset_path: str | None = None,
    max_eval_examples: int | None = None,
    generate_num: int | None = None,
):
    cfg = load_yaml(config_path)
    results_dir = repo_path(cfg.get("results_dir", "results/task1_dpo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset_file = eval_dataset_path or cfg["paths"]["dpo_standard_eval"]
    eval_rows = read_jsonl(dataset_file)
    if max_eval_examples is not None:
        eval_rows = eval_rows[:max_eval_examples]

    print(f"[{name}] Starting DPO evaluation:")
    print(f"  Adapter: {adapter_path}")
    print(f"  Dataset: {dataset_file} ({len(eval_rows)} examples)")

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=adapter_path, trainable=False)
    device = next(policy.parameters()).device

    # --- Part 1: Held-out Preference Metrics ---
    max_seq_len = int(cfg.get("max_sequence_length", 768))
    loader = DataLoader(
        eval_rows,
        batch_size=int(cfg.get("batch_size", 2)) * 2,
        shuffle=False,
        collate_fn=make_eval_collate(tokenizer, max_seq_len),
    )

    losses = []
    margins = []
    policy_margins = []
    ref_margins = []
    accuracies = []
    beta = float(cfg.get("beta", 0.10))

    with torch.no_grad():
        for chosen_batch, rejected_batch, _ in tqdm(loader, desc=f"Evaluating Pairs [{name}]"):
            if chosen_batch is None:
                continue
            chosen_batch = {k: v.to(device) for k, v in chosen_batch.items()}
            rejected_batch = {k: v.to(device) for k, v in rejected_batch.items()}

            with reference_mode(policy):
                ref_chosen_logp, _, _ = response_sequence_logprobs(policy, chosen_batch)
                ref_rejected_logp, _, _ = response_sequence_logprobs(policy, rejected_batch)

            pol_chosen_logp, _, _ = response_sequence_logprobs(policy, chosen_batch)
            pol_rejected_logp, _, _ = response_sequence_logprobs(policy, rejected_batch)

            loss, diag = dpo_loss(
                pol_chosen_logp,
                pol_rejected_logp,
                ref_chosen_logp,
                ref_rejected_logp,
                beta=beta,
            )

            pol_m = pol_chosen_logp - pol_rejected_logp
            ref_m = ref_chosen_logp - ref_rejected_logp
            m = pol_m - ref_m

            losses.append(loss.item())
            margins.extend(m.cpu().tolist())
            policy_margins.extend(pol_m.cpu().tolist())
            ref_margins.extend(ref_m.cpu().tolist())
            accuracies.append(diag["preference_accuracy"].item())

    mean_loss = float(np.mean(losses)) if losses else 0.0
    pref_acc = float(np.mean([1.0 if m > 0 else 0.0 for m in margins])) if margins else 0.0
    mean_margin = float(np.mean(margins)) if margins else 0.0

    # --- Part 2: Generations and Reward Model Scoring ---
    rm_model, rm_tok = load_reward_model(cfg)

    # Use all prompts for generation by default, or subset if generate_num specified
    gen_candidates = eval_rows if (generate_num is None or generate_num <= 0) else eval_rows[:generate_num]
    gen_prompts = [prompt_messages_from_preference(r) for r in gen_candidates]

    max_prompt_len = int(cfg.get("max_prompt_length", 256))
    max_gen_tokens = int(cfg.get("max_generation_tokens", 256))
    gen_bsz = 4

    all_generated_texts = []
    all_lengths = []
    all_kls = []
    all_entropies = []
    all_rewards = []
    records = []

    for start_idx in tqdm(range(0, len(gen_prompts), gen_bsz), desc=f"Generating [{name}]"):
        chunk_prompts = gen_prompts[start_idx : start_idx + gen_bsz]
        chunk_rows = gen_candidates[start_idx : start_idx + gen_bsz]

        gen_out = batch_generate(
            policy,
            tokenizer,
            chunk_prompts,
            max_prompt_length=max_prompt_len,
            max_new_tokens=max_gen_tokens,
            temperature=float(cfg.get("generation", {}).get("temperature", 0.7)),
            top_p=float(cfg.get("generation", {}).get("top_p", 0.9)),
            do_sample=bool(cfg.get("generation", {}).get("do_sample", True)),
        )

        # Compute token log-probs for policy and reference under no_grad
        with torch.no_grad():
            pol_tok_logp, _ = response_token_logprobs(
                policy,
                gen_out["sequences"],
                gen_out["attention_mask"],
                gen_out["prompt_width"],
                gen_out["response_ids"],
            )

            with reference_mode(policy):
                ref_tok_logp, _ = response_token_logprobs(
                    policy,
                    gen_out["sequences"],
                    gen_out["attention_mask"],
                    gen_out["prompt_width"],
                    gen_out["response_ids"],
                )

        # Compute per-response KL and entropy
        for b in range(len(chunk_prompts)):
            mask = gen_out["response_mask"][b]
            if mask.sum() > 0:
                p_lp = pol_tok_logp[b]
                r_lp = ref_tok_logp[b]
                kl_val = sampled_kl(p_lp, r_lp, mask).item()
                ent_val = sample_entropy(p_lp, mask).item()
            else:
                kl_val = 0.0
                ent_val = 0.0
            all_kls.append(kl_val)
            all_entropies.append(ent_val)

        # Reward model score
        rewards = score_reward_pairs(
            rm_model,
            rm_tok,
            chunk_prompts,
            gen_out["responses"],
            max_length=int(cfg.get("reward_max_length", 1024)),
        ).cpu().tolist()

        all_rewards.extend(rewards)
        all_lengths.extend(gen_out["response_lengths"])
        all_generated_texts.extend(gen_out["responses"])

        for r_row, prompt, text, n_tok, r_score, kl_v in zip(
            chunk_rows,
            chunk_prompts,
            gen_out["responses"],
            gen_out["response_lengths"],
            rewards,
            all_kls[-len(chunk_prompts):],
        ):
            records.append({
                "prompt": prompt,
                "response": text,
                "response_tokens": int(n_tok),
                "reward_score": round(float(r_score), 4),
                "kl": round(float(kl_v), 4),
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
        "dataset": str(dataset_file),
        "num_eval_pairs": len(eval_rows),
        "dpo_loss": round(mean_loss, 4),
        "preference_accuracy": round(pref_acc, 4),
        "mean_margin": round(mean_margin, 4),
        "num_generations": len(all_generated_texts),
        "reward_mean": round(float(np.mean(rewards_arr)), 4),
        "reward_std": round(float(np.std(rewards_arr)), 4),
        "kl_mean": round(float(np.mean(kl_arr)), 4),
        "kl_std": round(float(np.std(kl_arr)), 4),
        "entropy_mean": round(float(np.mean(ent_arr)), 4),
        "length_mean": round(float(np.mean(lengths_arr)), 2),
        "length_std": round(float(np.std(lengths_arr)), 2),
        "length_median": round(float(np.median(lengths_arr)), 2),
        "length_iqr": round(float(iqr), 2),
    }

    # Save outputs
    eval_json = results_dir / f"{name}_eval.json"
    with open(eval_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    gen_jsonl = results_dir / f"{name}_generations.jsonl"
    write_jsonl(gen_jsonl, records)

    print("\n" + "=" * 60)
    print(f"DPO Evaluation Summary: [{name}]")
    print(f"  Held-out Loss:        {metrics['dpo_loss']}")
    print(f"  Preference Accuracy:  {metrics['preference_accuracy'] * 100:.2f}%")
    print(f"  Mean Margin:          {metrics['mean_margin']}")
    print(f"  KL from Reference:    {metrics['kl_mean']:.4f} ± {metrics['kl_std']:.4f}")
    print(f"  Reward Model Score:   {metrics['reward_mean']:.4f} ± {metrics['reward_std']:.4f}")
    print(f"  Entropy:              {metrics['entropy_mean']:.4f}")
    print(f"  Response Length:      {metrics['length_mean']} ± {metrics['length_std']} tokens (IQR: {metrics['length_iqr']})")
    print("=" * 60)
    print(f"Results saved to: {eval_json}")
    print(f"Generations saved to: {gen_jsonl}\n")

    clear_gpu(policy, rm_model)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--eval-dataset")
    ap.add_argument("--max-eval-examples", type=int)
    ap.add_argument("--generate-num", type=int, default=None)
    args = ap.parse_args()
    evaluate_dpo(
        config_path=args.config,
        adapter_path=args.adapter,
        name=args.name,
        eval_dataset_path=args.eval_dataset,
        max_eval_examples=args.max_eval_examples,
        generate_num=args.generate_num,
    )


if __name__ == "__main__":
    main()
