from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from common.data import load_yaml, repo_path
from task1_dpo.evaluate import evaluate_dpo
from task1_dpo.train import run_training


def run_beta_ablation(
    config_path: str,
    generate_num: int | None = None,
    force_train: bool = False,
):
    cfg = load_yaml(config_path)
    betas = cfg.get("betas", [0.03, 0.10, 0.30])
    short_examples = int(cfg.get("short_ablation_examples", 600))
    results_dir = repo_path(cfg.get("results_dir", "results/task1_dpo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TASK 1: DPO REGULARIZATION-STRENGTH (BETA) ABLATION")
    print(f"Tested Betas: {betas}")
    print(f"Fixed short budget: {short_examples} examples per fork")
    print("=" * 60)

    all_metrics = []

    for b in betas:
        run_name = f"beta_{b}"
        out_adapter = f"outputs/task1_dpo/beta_{b}"
        adapter_path = repo_path(out_adapter)

        # Train if adapter does not exist or force_train is True
        if not (adapter_path / "adapter_config.json").exists() or force_train:
            print(f"\n>>> Running training for beta={b} ({run_name})")
            run_training(
                config_path=config_path,
                run_name=run_name,
                dataset_path=None,
                output_path=out_adapter,
                beta=b,
                max_examples=short_examples,
            )
        else:
            print(f"\n>>> Found existing adapter at {out_adapter}, skipping training.")

        # Evaluate
        print(f"\n>>> Running evaluation for {run_name}")
        metrics = evaluate_dpo(
            config_path=config_path,
            adapter_path=out_adapter,
            name=run_name,
            generate_num=generate_num,
        )
        metrics["beta"] = b
        metrics["budget"] = f"{short_examples} examples"
        all_metrics.append(metrics)

    # Check for standard DPO run
    std_eval_file = results_dir / "standard_eval.json"
    if std_eval_file.exists():
        with open(std_eval_file, "r", encoding="utf-8") as f:
            std_metrics = json.load(f)
            std_metrics["beta"] = cfg.get("beta", 0.10)
            std_metrics["budget"] = "1 full epoch"
            all_metrics.insert(0, std_metrics)

    # Convert to DataFrame for summary table
    df_rows = []
    for m in all_metrics:
        df_rows.append({
            "Run": m.get("run_name"),
            "Beta": m.get("beta"),
            "Budget": m.get("budget"),
            "Held-out Loss": m.get("dpo_loss"),
            "Pref Acc (%)": round(m.get("preference_accuracy", 0) * 100, 2),
            "KL Drift": m.get("kl_mean"),
            "Reward Score": m.get("reward_mean"),
            "Entropy": m.get("entropy_mean"),
            "Length (Mean±Std)": f"{m.get('length_mean')} ± {m.get('length_std')}",
            "Length IQR": m.get("length_iqr"),
        })

    summary_df = pd.DataFrame(df_rows)

    # Save summary
    summary_json = results_dir / "beta_ablation_summary.json"
    summary_csv = results_dir / "beta_ablation_summary.csv"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    summary_df.to_csv(summary_csv, index=False)

    print("\n" + "=" * 70)
    print("DPO BETA ABLATION COMPARISON SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70)
    print(f"Summary JSON saved to: {summary_json}")
    print(f"Summary CSV saved to:  {summary_csv}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--generate-num", type=int, default=None)
    ap.add_argument("--force-train", action="store_true")
    args = ap.parse_args()
    run_beta_ablation(
        config_path=args.config,
        generate_num=args.generate_num,
        force_train=args.force_train,
    )


if __name__ == "__main__":
    main()
