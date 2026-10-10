from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from task2_ppo.continue_train import run_ppo
from task2_ppo.evaluate import evaluate_ppo


def run_kl_ablation(config_path: str, max_eval_prompts: int = 32, force_train: bool = False):
    cfg = load_yaml(config_path)
    kl_values = [float(x) for x in cfg.get("kl_values", [0.0, 0.10, 0.20])]
    fork_updates = int(cfg.get("fork_updates", 8))
    clip_eps = float(cfg.get("clip_epsilon", 0.20))
    results_dir = repo_path(cfg.get("results_dir", "results/task2_ppo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TASK 2: PPO KL-PRESSURE (REWARD-OVEROPTIMIZATION) ABLATION")
    print(f"Tested KL Betas: {kl_values}")
    print(f"Fixed Short Budget: {fork_updates} updates per fork")
    print(f"Fixed Clipping Epsilon: {clip_eps}")
    print("=" * 70)

    all_metrics = []
    generations_by_beta = {}

    for b in kl_values:
        run_name = f"kl_{b}"
        out_adapter = f"outputs/task2_ppo/kl_{b}"
        adapter_path = repo_path(out_adapter)

        # Train if adapter does not exist or force_train
        if not (adapter_path / "adapter_config.json").exists() or force_train:
            print(f"\n>>> Running short PPO continuation for kl_beta={b} ({run_name})")
            run_ppo(
                config_path=config_path,
                output=out_adapter,
                updates=fork_updates,
                clip_epsilon=clip_eps,
                kl_beta=b,
                run_name=run_name,
            )
        else:
            print(f"\n>>> Found existing adapter at {out_adapter}, skipping training.")

        # Evaluate final policy on held-out prompts
        print(f">>> Evaluating held-out performance for {run_name}")
        metrics = evaluate_ppo(
            config_path=config_path,
            adapter_path=out_adapter,
            name=run_name,
            max_prompts=max_eval_prompts,
        )
        metrics["kl_beta"] = b
        all_metrics.append(metrics)

        # Collect generations for qualitative comparison
        gen_file = results_dir / f"{run_name}_generations.jsonl"
        if gen_file.exists():
            generations_by_beta[b] = read_jsonl(gen_file)

    # Convert to DataFrame
    df_rows = []
    for m in all_metrics:
        df_rows.append({
            "KL Beta": m["kl_beta"],
            "Held-out Reward": f"{m['reward_mean']} ± {m['reward_std']}",
            "Held-out KL": f"{m['kl_mean']} ± {m['kl_std']}",
            "Entropy": m["entropy_mean"],
            "Response Length": f"{m['length_mean']} ± {m['length_std']}",
            "Length IQR": m["length_iqr"],
            "EOS Rate (%)": round(m["eos_rate"] * 100, 1),
        })
    summary_df = pd.DataFrame(df_rows)

    # Identify qualitative examples of reward overoptimization (where KL=0 achieves high reward but diverges)
    qualitative_cases = []
    if 0.0 in generations_by_beta and 0.10 in generations_by_beta:
        b0_rows = generations_by_beta[0.0]
        b01_rows = generations_by_beta[0.10]
        for r0, r01 in zip(b0_rows[:10], b01_rows[:10]):
            qualitative_cases.append({
                "prompt": r0["prompt"],
                "beta_0.0_response": r0["response"],
                "beta_0.0_reward": r0["reward_score"],
                "beta_0.0_kl": r0["kl"],
                "beta_0.10_response": r01["response"],
                "beta_0.10_reward": r01["reward_score"],
                "beta_0.10_kl": r01["kl"],
            })

    # Save summary files
    out_json = results_dir / "kl_ablation_summary.json"
    out_csv = results_dir / "kl_ablation_summary.csv"
    qual_json = results_dir / "kl_ablation_qualitative.json"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    summary_df.to_csv(out_csv, index=False)
    with open(qual_json, "w", encoding="utf-8") as f:
        json.dump(qualitative_cases, f, indent=2)

    print("\n" + "=" * 80)
    print("PPO KL-PRESSURE ABLATION: SUMMARY COMPARISON")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)
    print(f"Saved summary JSON:     {out_json}")
    print(f"Saved summary CSV:      {out_csv}")
    print(f"Saved qualitative JSON: {qual_json}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--max-eval-prompts", type=int, default=32)
    ap.add_argument("--force-train", action="store_true")
    args = ap.parse_args()
    run_kl_ablation(
        config_path=args.config,
        max_eval_prompts=args.max_eval_prompts,
        force_train=args.force_train,
    )


if __name__ == "__main__":
    main()
