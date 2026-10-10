from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import (
    batch_generate,
    response_token_logprobs,
    score_reward_pairs,
)
from common.logging_utils import set_seed
from common.models import (
    clear_gpu,
    load_policy,
    load_reward_model,
    load_tokenizer,
    reference_mode,
    trainable_parameters,
)
from task3_grpo.grpo import (
    group_relative_advantages,
    grpo_policy_loss,
    mask_truncated_sequences,
)


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


def run_grpo(
    config_path: str,
    output: str | None = None,
    updates: int | None = None,
    loss_type: str = "grpo",
    run_name: str = "standard",
):
    start_time = time.time()
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]

    num_updates = int(updates if updates is not None else cfg.get("updates", 20))
    num_gens = int(cfg.get("num_generations", 4))
    clip_eps = float(cfg.get("clip_epsilon", 0.20))
    beta_kl = float(cfg.get("kl_beta", 0.10))

    policy = bundle["policy"]
    reward_model = bundle["reward_model"]
    reward_tokenizer = bundle["reward_tokenizer"]
    tokenizer = bundle["tokenizer"]
    prompts = bundle["prompt_rows"]
    optimizer = bundle["optimizer"]

    out_path = repo_path(output or cfg["output"])
    out_path.mkdir(parents=True, exist_ok=True)
    results_dir = repo_path(cfg.get("results_dir", "results/task3_grpo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    max_prompt_len = int(cfg.get("max_prompt_length", 256))
    max_comp_len = int(cfg.get("max_completion_length", 512))
    mask_truncated = bool(cfg.get("mask_truncated_completions", True))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))

    device = next(policy.parameters()).device

    print(f"[{run_name}] Starting GRPO continuation:")
    print(f"  Updates: {num_updates}, Generations/prompt (K): {num_gens}")
    print(f"  Clip epsilon: {clip_eps}, KL beta: {beta_kl}, Loss type: {loss_type}")
    print(f"  Max completion length: {max_comp_len}, Mask truncated: {mask_truncated}")
    print(f"  Output path: {out_path}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    history = []
    pbar = tqdm(total=num_updates, desc=f"GRPO [{run_name}]")

    for update_idx in range(num_updates):
        # 1. Select prompt and create K identical prompt messages
        prompt_item = prompts[update_idx % len(prompts)]
        prompt_msgs = prompt_messages(prompt_item)
        prompts_k = [prompt_msgs] * num_gens

        # 2. On-policy Rollout Generation of K completions
        policy.eval()
        gen = batch_generate(
            policy,
            tokenizer,
            prompts_k,
            max_prompt_length=max_prompt_len,
            max_new_tokens=max_comp_len,
            temperature=float(cfg.get("generation", {}).get("temperature", 0.7)),
            top_p=float(cfg.get("generation", {}).get("top_p", 0.9)),
            do_sample=True,
        )

        resp_lengths = gen["response_lengths"]
        if sum(resp_lengths) == 0:
            pbar.update(1)
            continue

        # 3. Terminal learned reward scoring
        with torch.no_grad():
            raw_rewards = score_reward_pairs(
                reward_model,
                reward_tokenizer,
                prompts_k,
                gen["responses"],
                max_length=int(cfg.get("reward_max_length", 1280)),
            )

        reward_vals = raw_rewards.float()
        mean_reward = float(reward_vals.mean().item())
        group_reward_std = float(reward_vals.std(unbiased=False).item())
        is_uninformative = float(group_reward_std < 1e-4)

        # 4. Old-policy token log-probabilities
        policy.eval()
        with torch.no_grad():
            old_logp, _ = response_token_logprobs(
                policy,
                gen["sequences"],
                gen["attention_mask"],
                gen["prompt_width"],
                gen["response_ids"],
            )

        # 5. Reference token log-probabilities
        with torch.no_grad():
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(
                    policy,
                    gen["sequences"],
                    gen["attention_mask"],
                    gen["prompt_width"],
                    gen["response_ids"],
                )

        # 6. Apply truncation masking if configured
        token_mask = gen["response_mask"]
        if mask_truncated:
            token_mask = mask_truncated_sequences(token_mask, gen["truncated"])

        # 7. Group-relative advantages
        group_ids = torch.zeros(num_gens, device=device, dtype=torch.long)
        advantages = group_relative_advantages(reward_vals, group_ids)

        # 8. Policy forward pass and optimization
        policy.train()
        new_logp, _ = response_token_logprobs(
            policy,
            gen["sequences"],
            gen["attention_mask"],
            gen["prompt_width"],
            gen["response_ids"],
        )

        loss, diag = grpo_policy_loss(
            new_logp,
            old_logp,
            advantages,
            token_mask,
            ref_logp,
            eps=clip_eps,
            beta=beta_kl,
            loss_type=loss_type,
            max_completion_length=max_comp_len,
        )

        optimizer.zero_grad()
        p_norm_val = 0.0
        if not torch.isnan(loss) and not torch.isinf(loss):
            loss.backward()
            p_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_grad_norm)
            p_norm_val = float(p_norm.item() if hasattr(p_norm, "item") else p_norm)
            if not (torch.isnan(p_norm) or torch.isinf(p_norm)):
                optimizer.step()
            else:
                optimizer.zero_grad()
        else:
            optimizer.zero_grad()

        elapsed = time.time() - start_time
        peak_vram = round(torch.cuda.max_memory_allocated() / (1024**3), 2) if torch.cuda.is_available() else 0.0

        step_record = {
            "update": update_idx + 1,
            "reward": round(mean_reward, 4),
            "group_reward_std": round(group_reward_std, 4),
            "uninformative_group": round(is_uninformative, 4),
            "kl": round(float(diag["sampled_kl"].item()), 4),
            "policy_loss": round(float(loss.item()), 4),
            "gradient_norm": round(p_norm_val, 4),
            "entropy": round(float(diag["sample_entropy"].item()), 4),
            "clip_fraction": round(float(diag["clip_fraction"].item()), 4),
            "mean_response_length": round(float(np.mean(resp_lengths)), 2),
            "wall_clock_time": round(elapsed, 1),
            "peak_vram_gib": peak_vram,
        }
        history.append(step_record)

        # Persist intermediate logs immediately
        log_path = results_dir / f"{run_name}_train_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        pbar.set_postfix({
            "R": f"{mean_reward:.2f}",
            "R_std": f"{group_reward_std:.2f}",
            "KL": f"{float(diag['sampled_kl'].item()):.3f}",
            "P_loss": f"{float(loss.item()):.3f}",
            "Len": f"{np.mean(resp_lengths):.0f}",
        })
        pbar.update(1)

    pbar.close()
    total_time = time.time() - start_time

    # Save trained policy adapter
    print(f"\n[{run_name}] Saving trained GRPO adapter to {out_path}...")
    policy.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))

    peak_vram = round(torch.cuda.max_memory_allocated() / (1024**3), 2) if torch.cuda.is_available() else 0.0

    print(f"[{run_name}] GRPO continuation finished:")
    print(f"  Wall-clock time: {total_time:.1f}s ({total_time / 60:.2f} min)")
    print(f"  Peak VRAM: {peak_vram} GiB")
    print(f"  Training log: {log_path}\n")

    clear_gpu(policy, reward_model)
    return {
        "output_path": str(out_path),
        "log_path": str(log_path),
        "wall_clock_time": total_time,
        "peak_vram_gib": peak_vram,
        "history": history,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name)


if __name__ == "__main__":
    main()
