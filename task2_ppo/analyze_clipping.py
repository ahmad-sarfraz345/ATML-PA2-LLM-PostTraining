from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.data import load_yaml, repo_path
from common.metrics import masked_mean
from common.models import clear_gpu, load_policy, load_tokenizer
from task2_ppo.continue_train import run_ppo
from task2_ppo.evaluate import evaluate_ppo


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def evaluate_cached_clipping_geometry(cfg, rows, clip_epsilons: list[float]):
    print(f"\nEvaluating geometric clipping on {len(rows)} cached rollouts...")

    # Check if advantages or returns are already present in cache
    has_adv = "advantages" in rows[0] or "advantage" in rows[0]
    cached_metrics = {}

    for eps in clip_epsilons:
        affected_tokens = 0
        total_tokens = 0
        surrogate_vals = []

        for row in rows:
            old_logp = torch.as_tensor(row["old_logprobs"], dtype=torch.float32)
            ref_logp = torch.as_tensor(row["ref_logprobs"], dtype=torch.float32)

            # In the absence of an updated policy, compare old vs ref or compute self-ratio
            # If advantages exist, compute full surrogate; otherwise evaluate importance ratio distribution
            ratio = torch.exp(old_logp - ref_logp) if "new_logprobs" not in row else torch.exp(torch.as_tensor(row["new_logprobs"]) - old_logp)

            # Affected token condition: ratio outside [1 - eps, 1 + eps]
            is_affected = (ratio < (1.0 - eps)) | (ratio > (1.0 + eps))
            affected_tokens += int(is_affected.sum().item())
            total_tokens += int(ratio.numel())

            if has_adv:
                adv = torch.as_tensor(row.get("advantages", row.get("advantage")), dtype=torch.float32)
                s1 = ratio * adv
                s2 = ratio.clamp(1.0 - eps, 1.0 + eps) * adv
                surr = torch.minimum(s1, s2).mean().item()
                surrogate_vals.append(surr)

        affected_frac = affected_tokens / max(total_tokens, 1)
        mean_surr = float(np.mean(surrogate_vals)) if surrogate_vals else 0.0

        cached_metrics[eps] = {
            "clip_epsilon": eps,
            "affected_tokens": affected_tokens,
            "total_tokens": total_tokens,
            "affected_token_fraction": round(affected_frac, 4),
            "clipped_surrogate_mean": round(mean_surr, 4),
        }

    return cached_metrics


def run_clipping_analysis(config_path: str, max_eval_prompts: int = 32, force_train: bool = False):
    cfg = load_yaml(config_path)
    clip_values = [float(x) for x in cfg.get("clip_values", [0.05, 0.20, 0.50])]
    fork_updates = int(cfg.get("fork_updates", 8))
    results_dir = repo_path(cfg.get("results_dir", "results/task2_ppo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TASK 2: PPO CLIPPING STUDY")
    print(f"Tested Epsilons: {clip_values}")
    print(f"Continuation Budget: {fork_updates} updates per fork")
    print("=" * 70)

    # 1. Immediate Geometric Effect on Cached Rollout Batch
    cached_rows = load_cached_rollouts(cfg["cached_rollouts"])
    cached_results = evaluate_cached_clipping_geometry(cfg, cached_rows, clip_values)

    # 2. Run Matched Short Continuation Forks
    fork_summaries = []

    for eps in clip_values:
        run_name = f"clip_{eps}"
        out_adapter = f"outputs/task2_ppo/clip_{eps}"
        adapter_path = repo_path(out_adapter)

        # Train if adapter does not exist or force_train
        if not (adapter_path / "adapter_config.json").exists() or force_train:
            print(f"\n>>> Running short PPO continuation for clip_epsilon={eps} ({run_name})")
            run_ppo(
                config_path=config_path,
                output=out_adapter,
                updates=fork_updates,
                clip_epsilon=eps,
                kl_beta=float(cfg.get("kl_beta", 0.10)),
                run_name=run_name,
            )
        else:
            print(f"\n>>> Found existing adapter at {out_adapter}, skipping training.")

        # Read training log to compute stability statistics (Option A: loss & grad variances)
        train_log_file = results_dir / f"{run_name}_train_log.json"
        pol_loss_var = 0.0
        grad_norm_var = 0.0
        grad_norm_mean = 0.0
        mean_clip_frac = 0.0

        if train_log_file.exists():
            with open(train_log_file, "r", encoding="utf-8") as f:
                log_data = json.load(f)
            losses = [x["policy_loss"] for x in log_data]
            grads = [x["gradient_norm"] for x in log_data]
            clips = [x["clip_fraction"] for x in log_data]
            pol_loss_var = float(np.var(losses)) if losses else 0.0
            grad_norm_var = float(np.var(grads)) if grads else 0.0
            grad_norm_mean = float(np.mean(grads)) if grads else 0.0
            mean_clip_frac = float(np.mean(clips)) if clips else 0.0

        # Evaluate final policy on held-out prompts
        print(f">>> Evaluating held-out performance for {run_name}")
        eval_metrics = evaluate_ppo(
            config_path=config_path,
            adapter_path=out_adapter,
            name=run_name,
            max_prompts=max_eval_prompts,
        )

        fork_summaries.append({
            "Epsilon": eps,
            "Cached Affected Frac": cached_results[eps]["affected_token_fraction"],
            "Held-out Reward": f"{eval_metrics['reward_mean']} ± {eval_metrics['reward_std']}",
            "Held-out KL": f"{eval_metrics['kl_mean']} ± {eval_metrics['kl_std']}",
            "Response Length": f"{eval_metrics['length_mean']} ± {eval_metrics['length_std']}",
            "Training Mean Clip Frac": round(mean_clip_frac, 4),
            "Policy Loss Var (Stability)": round(pol_loss_var, 4),
            "Grad Norm Var (Stability)": round(grad_norm_var, 4),
            "Grad Norm Mean": round(grad_norm_mean, 4),
            "raw_eval": eval_metrics,
        })

    summary_df = pd.DataFrame(fork_summaries)

    # Save outputs
    out_json = results_dir / "clipping_analysis.json"
    out_csv = results_dir / "clipping_analysis.csv"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "cached_batch_geometry": cached_results,
            "fork_summaries": fork_summaries,
        }, f, indent=2)

    summary_df.drop(columns=["raw_eval"]).to_csv(out_csv, index=False)

    print("\n" + "=" * 80)
    print("PPO CLIPPING STUDY: SUMMARY COMPARISON")
    print("=" * 80)
    print(summary_df.drop(columns=["raw_eval"]).to_string(index=False))
    print("=" * 80)
    print(f"Saved analysis JSON: {out_json}")
    print(f"Saved analysis CSV:  {out_csv}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--max-eval-prompts", type=int, default=32)
    ap.add_argument("--force-train", action="store_true")
    args = ap.parse_args()
    run_clipping_analysis(
        config_path=args.config,
        max_eval_prompts=args.max_eval_prompts,
        force_train=args.force_train,
    )


if __name__ == "__main__":
    main()
