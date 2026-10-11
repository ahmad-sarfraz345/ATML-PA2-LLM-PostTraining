from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from common.data import load_yaml, repo_path
from task5_feedback.evaluate_math import evaluate_dataset
from task5_feedback.score_perturbations import score_diagnostics


def compare_all_feedback(config_path: str = "configs/feedback.yaml", run_missing: bool = True):
    cfg = load_yaml(config_path)
    res_dir = repo_path(cfg.get("results_dir", "results")) / "task5_feedback"
    res_dir.mkdir(parents=True, exist_ok=True)

    gsm_json = res_dir / "gsm_evaluation_summary.json"
    transfer_json = res_dir / "transfer_evaluation_summary.json"
    perturb_json = res_dir / "perturbation_scores.json"

    # Run any missing stages if run_missing is True
    if not gsm_json.exists():
        if run_missing:
            print("[compare_feedback] In-domain GSM8K evaluation not found. Running now...")
            evaluate_dataset(config_path=config_path, dataset="gsm")
        else:
            raise FileNotFoundError(f"Missing {gsm_json}. Run evaluate_math with --dataset gsm first.")

    if not transfer_json.exists():
        if run_missing:
            print("[compare_feedback] Out-of-domain SVAMP transfer evaluation not found. Running now...")
            evaluate_dataset(config_path=config_path, dataset="transfer")
        else:
            raise FileNotFoundError(f"Missing {transfer_json}. Run evaluate_math with --dataset transfer first.")

    if not perturb_json.exists():
        if run_missing:
            print("[compare_feedback] Controlled perturbation study not found. Running now...")
            score_diagnostics(config_path=config_path)
        else:
            raise FileNotFoundError(f"Missing {perturb_json}. Run score_perturbations first.")

    with open(gsm_json, "r", encoding="utf-8") as f:
        gsm_data = json.load(f)
    with open(transfer_json, "r", encoding="utf-8") as f:
        transfer_data = json.load(f)
    with open(perturb_json, "r", encoding="utf-8") as f:
        perturb_data = json.load(f)

    print("=" * 80)
    print("TASK 5: RLVR VS RLAIF SYNTHESIS & DOMAIN TRANSFER ANALYSIS")
    print("=" * 80)

    policies = ["sft", "rlvr", "rlaif"]
    rows = []

    for p in policies:
        gsm_p = gsm_data["policies"][p]
        trans_p = transfer_data["policies"][p]

        in_acc = gsm_p["accuracy"] * 100
        trans_acc = trans_p["accuracy"] * 100
        acc_drop = in_acc - trans_acc

        in_win = 50.0
        trans_win = 50.0
        if p != "sft":
            pair_key = f"{p}_vs_sft"
            in_win = gsm_data["pairwise"][pair_key]["win_rate_a"] * 100
            trans_win = transfer_data["pairwise"][pair_key]["win_rate_a"] * 100
        win_drop = in_win - trans_win

        rows.append({
            "Policy": p.upper(),
            "In-Domain Acc (%)": round(in_acc, 2),
            "Transfer Acc (%)": round(trans_acc, 2),
            "Acc Drop (%)": round(acc_drop, 2),
            "In-Domain WinRate vs SFT (%)": round(in_win, 2),
            "Transfer WinRate vs SFT (%)": round(trans_win, 2),
            "WinRate Drop (%)": round(win_drop, 2),
            "In-Domain Format (%)": round(gsm_p["format_compliance"] * 100, 2),
            "In-Domain Mean Tokens": round(gsm_p["mean_length"], 1),
            "Transfer Mean Tokens": round(trans_p["mean_length"], 1),
        })

    summary_df = pd.DataFrame(rows)
    print("\nIN-DOMAIN VS OUT-OF-DOMAIN TRANSFER COMPARISON TABLE:")
    print(summary_df.to_string(index=False))

    # Sensitivity comparison
    print("\n" + "=" * 80)
    print("REWARD SENSITIVITY & ROBUSTNESS SYNTHESIS:")
    print("=" * 80)
    s_reason = perturb_data["s_reason"]
    s_outcome = perturb_data["s_outcome"]
    print(f"  Reasoning Sensitivity (S_reason = Pr[Clean > Corrupt]):")
    print(f"    - Exact Verifier (RLVR): {s_reason['verifier'] * 100:.1f}% (Invariance to reasoning when answer is correct)")
    print(f"    - AI Judge (RLAIF):      {s_reason['judge'] * 100:.1f}% (Distinguishes reasoning steps)")

    print(f"\n  Outcome Sensitivity (S_outcome = Pr[Clean > Wrong]):")
    print(f"    - Exact Verifier (RLVR): {s_outcome['verifier'] * 100:.1f}% (Strict outcome enforcement)")
    print(f"    - AI Judge (RLAIF):      {s_outcome['judge'] * 100:.1f}% (Penalizes wrong answer despite coherent steps)")

    # Qualitative Disagreements
    cases_file = res_dir / "qualitative_diagnostic_cases.json"
    cases = []
    if cases_file.exists():
        with open(cases_file, "r", encoding="utf-8") as f:
            cases = json.load(f)
        print(f"\nDiscovered {len(cases)} qualitative disagreement case(s) in diagnostic study.")

    # Save final artifacts
    out_csv = res_dir / "final_comparison_table.csv"
    out_json = res_dir / "final_comparison_summary.json"

    summary_df.to_csv(out_csv, index=False)

    final_payload = {
        "domain_transfer_table": rows,
        "s_reason": s_reason,
        "s_outcome": s_outcome,
        "feedback_synthesis": {
            "rlvr": {
                "coverage": "Exact binary outcome on designated final answer (sparse).",
                "noise": "Zero preference noise; invariant to reasoning steps.",
                "exploitability": "Vulnerable to reward hacking via corrupted reasoning ending in correct answer.",
                "inference_cost": "Negligible (regex and numeric comparison).",
            },
            "rlaif": {
                "coverage": "Continuous relative win rate over solution pairs (dense).",
                "noise": "Subject to LLM judge stochasticity, orientation bias, and framing artifacts.",
                "exploitability": "Susceptible to persuasive filler, style, and superficial verbosity.",
                "inference_cost": "High (requires 3B LLM chat template generation per compared pair).",
            },
        },
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    print(f"\nWrote final comparison table to {out_csv}")
    print(f"Wrote final comparison summary to {out_json}\n")
    return final_payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--no-run-missing", action="store_true", help="Do not run missing stages automatically")
    args = ap.parse_args()
    compare_all_feedback(config_path=args.config, run_missing=not args.no_run_missing)


if __name__ == "__main__":
    main()
