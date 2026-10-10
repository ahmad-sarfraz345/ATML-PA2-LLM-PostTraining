from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from common.data import load_yaml, read_jsonl, repo_path


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Validate each prompt has at least 8 completions
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int):
    """Partition the fixed 192 cached completions into K-sized groups.

    Total completions are held strictly constant at 192:
      K=8: 24 groups of 8 completions (1 group per prompt)
      K=4: 48 groups of 4 completions (2 sub-groups per prompt: [0..3], [4..7])
      K=2: 96 groups of 2 completions (4 sub-groups per prompt: [0..1], [2..3], [4..5], [6..7])
    """
    groups = []
    for pid, completions in by_prompt.items():
        # completions contains 8 rows
        n = len(completions)
        for i in range(0, n, k):
            sub = completions[i : i + k]
            if len(sub) == k:
                groups.append({
                    "prompt_id": pid,
                    "completions": sub,
                    "rewards": [float(r["reward"]) for r in sub],
                })
    return groups


def compute_group_statistics(groups):
    """Compute informative-group fraction, reward std, and relative advantage variance."""
    if not groups:
        return {
            "num_groups": 0,
            "informative_group_rate": 0.0,
            "mean_within_group_std": 0.0,
            "advantage_variance": 0.0,
            "mean_reward": 0.0,
        }

    stds = []
    informative_count = 0
    all_advantages = []
    all_rewards = []

    for g in groups:
        r_arr = np.array(g["rewards"], dtype=float)
        all_rewards.extend(g["rewards"])
        std_val = float(np.std(r_arr))
        stds.append(std_val)
        if std_val > 1e-4:
            informative_count += 1

        # Relative advantages within this group: (r - mean) / std
        if len(r_arr) <= 1:
            adv = np.zeros_like(r_arr)
        else:
            mean_val = float(np.mean(r_arr))
            clamped_std = max(std_val, 1e-6)
            adv = (r_arr - mean_val) / clamped_std
        all_advantages.extend(adv.tolist())

    inf_rate = informative_count / len(groups)
    mean_std = float(np.mean(stds))
    adv_var = float(np.var(all_advantages)) if all_advantages else 0.0
    mean_r = float(np.mean(all_rewards)) if all_rewards else 0.0

    return {
        "num_groups": len(groups),
        "informative_group_rate": round(inf_rate, 4),
        "mean_within_group_std": round(mean_std, 4),
        "advantage_variance": round(adv_var, 4),
        "mean_reward": round(mean_r, 4),
    }


def run_group_size_analysis(config_path: str):
    cfg = load_yaml(config_path)
    group_sizes = [int(k) for k in cfg.get("group_sizes", [2, 4, 8])]
    results_dir = repo_path(cfg.get("results_dir", "results/task3_grpo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    cache_path = repo_path(cfg["group_cache"])
    by_prompt = load_k8_cache(cache_path)
    num_prompts = len(by_prompt)

    print("=" * 70)
    print("TASK 3: GRPO EQUAL-GENERATION GROUP-SIZE STUDY")
    print(f"Cached Prompts: {num_prompts}, Total Completions: {num_prompts * 8}")
    print(f"Tested Group Sizes (K): {group_sizes}")
    print("=" * 70)

    # Define prompt difficulty binning:
    # Compute mean reward per prompt across all 8 cached completions.
    # Split into 2 bins: Hard (mean <= median) and Easy (mean > median).
    prompt_mean_rewards = {
        pid: float(np.mean([float(r["reward"]) for r in comps]))
        for pid, comps in by_prompt.items()
    }
    median_reward = float(np.median(list(prompt_mean_rewards.values())))
    print(f"\nPrompt Difficulty Binning (Median Cached Reward = {median_reward:.4f}):")
    hard_pids = {pid for pid, r in prompt_mean_rewards.items() if r <= median_reward}
    easy_pids = {pid for pid, r in prompt_mean_rewards.items() if r > median_reward}
    print(f"  Hard / Low-Reward Prompts: {len(hard_pids)}")
    print(f"  Easy / High-Reward Prompts: {len(easy_pids)}\n")

    overall_results = {}
    difficulty_results = {"Hard": {}, "Easy": {}}
    table_rows = []

    for k in group_sizes:
        groups_all = regroup_equal_generation_budget(by_prompt, k)
        groups_hard = [g for g in groups_all if g["prompt_id"] in hard_pids]
        groups_easy = [g for g in groups_all if g["prompt_id"] in easy_pids]

        stats_all = compute_group_statistics(groups_all)
        stats_hard = compute_group_statistics(groups_hard)
        stats_easy = compute_group_statistics(groups_easy)

        overall_results[k] = stats_all
        difficulty_results["Hard"][k] = stats_hard
        difficulty_results["Easy"][k] = stats_easy

        table_rows.append({
            "K (Group Size)": k,
            "Total Completions": k * len(groups_all),
            "Total Groups": len(groups_all),
            "Overall Informative Rate (%)": round(stats_all["informative_group_rate"] * 100, 1),
            "Overall Reward Std": stats_all["mean_within_group_std"],
            "Overall Adv Variance": stats_all["advantage_variance"],
            "Hard Informative Rate (%)": round(stats_hard["informative_group_rate"] * 100, 1),
            "Hard Adv Variance": stats_hard["advantage_variance"],
            "Easy Informative Rate (%)": round(stats_easy["informative_group_rate"] * 100, 1),
            "Easy Adv Variance": stats_easy["advantage_variance"],
        })

    summary_df = pd.DataFrame(table_rows)

    out_json = results_dir / "group_size_analysis.json"
    out_csv = results_dir / "group_size_analysis.csv"

    output_payload = {
        "group_sizes": group_sizes,
        "total_cached_completions": num_prompts * 8,
        "median_reward_threshold": median_reward,
        "overall": overall_results,
        "difficulty_breakdown": difficulty_results,
        "table": table_rows,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    summary_df.to_csv(out_csv, index=False)

    print("=" * 80)
    print("EQUAL-GENERATION GROUP-SIZE COMPARISON SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print(f"\nSaved analysis results to:\n  - {out_json}\n  - {out_csv}\n")
    return output_payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    run_group_size_analysis(args.config)


if __name__ == "__main__":
    main()
