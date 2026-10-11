from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from common.data import load_yaml, read_jsonl, repo_path
from common.models import clear_gpu
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def evaluate_pair(problem_text: str, gold_final: str, resp_better: str, resp_worse: str, judge: PairwiseAIJudge):
    # 1. Exact Verifier (RLVR)
    v_better = exact_reward(resp_better, gold_final)
    v_worse = exact_reward(resp_worse, gold_final)
    if v_better > v_worse:
        v_decision = "BETTER"
    elif v_better < v_worse:
        v_decision = "WORSE"
    else:
        v_decision = "TIE"

    # 2. Pairwise AI Judge (RLAIF)
    j_pref = judge.compare(problem_text, resp_better, resp_worse)
    if j_pref == "A":
        j_decision = "BETTER"
    elif j_pref == "B":
        j_decision = "WORSE"
    else:
        j_decision = "TIE"

    return {
        "verifier_decision": v_decision,
        "verifier_score_better": v_better,
        "verifier_score_worse": v_worse,
        "judge_decision": j_decision,
        "judge_raw_preference": j_pref,
    }


def score_diagnostics(config_path: str = "configs/feedback.yaml"):
    cfg = load_yaml(config_path)
    out_dir = repo_path(cfg.get("results_dir", "results")) / "task5_feedback"
    out_dir.mkdir(parents=True, exist_ok=True)

    diag_path = repo_path(cfg["paths"]["task5_diagnostics"])
    groups = load_diagnostic_groups(diag_path)
    print("=" * 80)
    print(f"TASK 5: CONTROLLED REWARD SENSITIVITY & ROBUSTNESS STUDY")
    print(f"Loaded {len(groups)} diagnostic problems (5 variants each = {len(groups)*5} responses)")
    print("=" * 80)

    judge_cache_file = out_dir / "ai_judge_cache.json"
    judge = PairwiseAIJudge(cfg, cache_path=judge_cache_file)

    # Define the 5 canonical perturbation categories to test:
    # Format: (Category Name, Better Variant, Worse Variant, Description)
    categories = [
        (
            "reasoning_corruption",
            "clean_correct",
            "corrupt_reasoning_correct_final",
            "Reasoning Sensitivity: same correct final answer, corrupted reasoning steps",
        ),
        (
            "outcome_corruption",
            "clean_correct",
            "good_reasoning_wrong_final",
            "Outcome Sensitivity: coherent reasoning, but wrong final answer",
        ),
        (
            "persuasive_filler",
            "clean_correct",
            "persuasive_filler_correct",
            "Filler Susceptibility: same correct answer & reasoning, but irrelevant filler added",
        ),
        (
            "gold_distractor",
            "clean_correct",
            "gold_distractor_wrong_final",
            "Distractor Robustness: gold number mentioned as distractor, wrong final answer",
        ),
        (
            "reasoning_vs_outcome",
            "corrupt_reasoning_correct_final",
            "good_reasoning_wrong_final",
            "Trade-off: correct answer with bad reasoning vs wrong answer with good reasoning",
        ),
    ]

    category_results = {}
    table_rows = []
    qualitative_diagnostic_samples = []

    for cat_key, v_better_key, v_worse_key, desc in categories:
        v_counts = {"BETTER": 0, "TIE": 0, "WORSE": 0}
        j_counts = {"BETTER": 0, "TIE": 0, "WORSE": 0}
        pair_details = []

        for pid, variants in tqdm(groups.items(), desc=f"Scoring {cat_key}"):
            var_better = variants[v_better_key]
            var_worse = variants[v_worse_key]

            problem = var_better.get("problem", var_better.get("question", ""))
            gold = str(var_better.get("gold_final", ""))

            resp_better = var_better["response"]
            resp_worse = var_worse["response"]

            eval_res = evaluate_pair(problem, gold, resp_better, resp_worse, judge)
            v_counts[eval_res["verifier_decision"]] += 1
            j_counts[eval_res["judge_decision"]] += 1

            pair_info = {
                "problem_id": pid,
                "problem": problem,
                "gold_final": gold,
                "better_variant": v_better_key,
                "worse_variant": v_worse_key,
                "verifier_decision": eval_res["verifier_decision"],
                "judge_decision": eval_res["judge_decision"],
                "response_better": resp_better,
                "response_worse": resp_worse,
            }
            pair_details.append(pair_info)

            # Collect representative qualitative examples of disagreement
            if eval_res["verifier_decision"] != eval_res["judge_decision"] and len(qualitative_diagnostic_samples) < 10:
                qualitative_diagnostic_samples.append({
                    "category": cat_key,
                    "problem_id": pid,
                    "problem": problem[:200],
                    "gold_final": gold,
                    "verifier_prefers": eval_res["verifier_decision"],
                    "judge_prefers": eval_res["judge_decision"],
                    "response_better_snippet": resp_better[:250],
                    "response_worse_snippet": resp_worse[:250],
                })

        n_p = len(groups)
        cat_data = {
            "description": desc,
            "total_pairs": n_p,
            "verifier": {
                "better_rate": round(v_counts["BETTER"] / n_p, 4),
                "tie_rate": round(v_counts["TIE"] / n_p, 4),
                "wrong_rate": round(v_counts["WORSE"] / n_p, 4),
            },
            "judge": {
                "better_rate": round(j_counts["BETTER"] / n_p, 4),
                "tie_rate": round(j_counts["TIE"] / n_p, 4),
                "wrong_rate": round(j_counts["WORSE"] / n_p, 4),
            },
            "pairs": pair_details,
        }
        category_results[cat_key] = cat_data

        # Add to summary table rows for both Verifier and Judge
        table_rows.append({
            "Perturbation Category": cat_key,
            "Mechanism": "Exact Verifier (RLVR)",
            "Better %": round(cat_data["verifier"]["better_rate"] * 100, 1),
            "Tie %": round(cat_data["verifier"]["tie_rate"] * 100, 1),
            "Wrong %": round(cat_data["verifier"]["wrong_rate"] * 100, 1),
        })
        table_rows.append({
            "Perturbation Category": cat_key,
            "Mechanism": "Pairwise Judge (RLAIF)",
            "Better %": round(cat_data["judge"]["better_rate"] * 100, 1),
            "Tie %": round(cat_data["judge"]["tie_rate"] * 100, 1),
            "Wrong %": round(cat_data["judge"]["wrong_rate"] * 100, 1),
        })

    clear_gpu(judge.model)

    # Compute S_reason and S_outcome
    s_reason_verifier = category_results["reasoning_corruption"]["verifier"]["better_rate"]
    s_reason_judge = category_results["reasoning_corruption"]["judge"]["better_rate"]

    s_outcome_verifier = category_results["outcome_corruption"]["verifier"]["better_rate"]
    s_outcome_judge = category_results["outcome_corruption"]["judge"]["better_rate"]

    summary_table_df = pd.DataFrame(table_rows)

    print("\n" + "=" * 80)
    print("CONTROLLED PERTURBATION STUDY RESULTS:")
    print("=" * 80)
    print(summary_table_df.to_string(index=False))

    print("\n" + "-" * 80)
    print(f"REASONING & OUTCOME SENSITIVITY METRICS:")
    print(f"  S_reason  (Pr[Clean > Corrupt]):  Verifier={s_reason_verifier:.4f} | AI Judge={s_reason_judge:.4f}")
    print(f"  S_outcome (Pr[Clean > Wrong]):    Verifier={s_outcome_verifier:.4f} | AI Judge={s_outcome_judge:.4f}")
    print("-" * 80)

    # Save to disk
    out_csv = out_dir / "perturbation_scores.csv"
    out_json = out_dir / "perturbation_scores.json"
    out_cases = out_dir / "qualitative_diagnostic_cases.json"

    summary_table_df.to_csv(out_csv, index=False)

    full_report = {
        "s_reason": {
            "verifier": s_reason_verifier,
            "judge": s_reason_judge,
        },
        "s_outcome": {
            "verifier": s_outcome_verifier,
            "judge": s_outcome_judge,
        },
        "categories": {k: {m: v[m] for m in ["description", "total_pairs", "verifier", "judge"]} for k, v in category_results.items()},
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    with open(out_cases, "w", encoding="utf-8") as f:
        json.dump(qualitative_diagnostic_samples, f, indent=2)

    print(f"\nWrote results to {out_csv}, {out_json}, and {out_cases}\n")
    return full_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    score_diagnostics(args.config)


if __name__ == "__main__":
    main()
