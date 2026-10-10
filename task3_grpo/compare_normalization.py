from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.data import load_yaml, repo_path
from common.metrics import safe_corr
from task3_grpo.continue_train import run_grpo
from task3_grpo.evaluate import evaluate_grpo


def run_normalization_comparison(
    config_path: str,
    max_eval_prompts: int = 32,
    force_train: bool = False,
):
    cfg = load_yaml(config_path)
    fork_updates = int(cfg.get("fork_updates", 8))
    results_dir = repo_path(cfg.get("results_dir", "results/task3_grpo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TASK 3: GRPO SEQUENCE-LENGTH NORMALIZATION STUDY")
    print(f"Continuation Budget: {fork_updates} updates per fork")
    print("Comparing: Canonical GRPO (1/T_k) vs Dr. GRPO (1/L, L=512)")
    print("=" * 70)

    conditions = [
        {"name": "grpo_canonical", "loss_type": "grpo", "desc": "Canonical GRPO (1/T_k)"},
        {"name": "grpo_dr", "loss_type": "dr_grpo", "desc": "Dr. GRPO (1/L constant)"},
    ]

    fork_summaries = []

    for cond in conditions:
        run_name = cond["name"]
        loss_type = cond["loss_type"]
        out_adapter = f"outputs/task3_grpo/{run_name}"
        adapter_path = repo_path(out_adapter)

        # Train if adapter does not exist or force_train
        if not (adapter_path / "adapter_config.json").exists() or force_train:
            print(f"\n>>> Running short GRPO continuation for {cond['desc']} ({run_name})")
            run_grpo(
                config_path=config_path,
                output=out_adapter,
                updates=fork_updates,
                loss_type=loss_type,
                run_name=run_name,
            )
        else:
            print(f"\n>>> Found existing adapter at {out_adapter}, skipping training.")

        # Read training log to extract length-conditioned gradient statistics
        train_log_file = results_dir / f"{run_name}_train_log.json"
        length_grad_corr = 0.0
        mean_grad_norm = 0.0
        mean_train_len = 0.0

        if train_log_file.exists():
            with open(train_log_file, "r", encoding="utf-8") as f:
                log_data = json.load(f)
            lens = [float(x.get("mean_response_length", 0)) for x in log_data]
            grads = [float(x.get("gradient_norm", 0)) for x in log_data]
            if len(lens) >= 2 and np.std(lens) > 0 and np.std(grads) > 0:
                length_grad_corr = round(float(safe_corr(lens, grads)), 4)
            mean_grad_norm = round(float(np.mean(grads)), 4) if grads else 0.0
            mean_train_len = round(float(np.mean(lens)), 2) if lens else 0.0

        # Evaluate final policy on held-out prompts
        print(f">>> Evaluating held-out performance for {run_name}")
        eval_metrics = evaluate_grpo(
            config_path=config_path,
            adapter_path=out_adapter,
            name=run_name,
            max_prompts=max_eval_prompts,
        )

        fork_summaries.append({
            "Normalization": cond["desc"],
            "Loss Type": loss_type,
            "Held-out Reward": f"{eval_metrics['reward_mean']} ± {eval_metrics['reward_std']}",
            "Held-out KL": f"{eval_metrics['kl_mean']} ± {eval_metrics['kl_std']}",
            "Response Length": f"{eval_metrics['length_mean']} ± {eval_metrics['length_std']}",
            "Length IQR": eval_metrics["length_iqr"],
            "EOS Rate (%)": round(eval_metrics["eos_rate"] * 100, 1),
            "Length-Gradient Corr": length_grad_corr,
            "Mean Training Grad Norm": mean_grad_norm,
            "Mean Training Length": mean_train_len,
            "raw_eval": eval_metrics,
        })

    summary_df = pd.DataFrame(fork_summaries)

    out_json = results_dir / "normalization_comparison.json"
    out_csv = results_dir / "normalization_comparison.csv"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "fork_updates": fork_updates,
            "fork_summaries": fork_summaries,
        }, f, indent=2)

    summary_df.drop(columns=["raw_eval"]).to_csv(out_csv, index=False)

    print("\n" + "=" * 80)
    print("GRPO NORMALIZATION COMPARISON SUMMARY")
    print("=" * 80)
    print(summary_df.drop(columns=["raw_eval"]).to_string(index=False))
    print(f"\nSaved normalization analysis to:\n  - {out_json}\n  - {out_csv}\n")
    return fork_summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--max-eval-prompts", type=int, default=32)
    ap.add_argument("--force-train", action="store_true")
    args = ap.parse_args()
    run_normalization_comparison(
        config_path=args.config,
        max_eval_prompts=args.max_eval_prompts,
        force_train=args.force_train,
    )


if __name__ == "__main__":
    main()
