from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.models import clear_gpu, load_policy, load_tokenizer
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(f"Unknown dataset: {dataset}")


def find_adapter_path(adapter: str | None) -> Path | None:
    if not adapter:
        return None
    p = repo_path(adapter)
    if p.exists():
        return p
    cwd_p = Path(adapter)
    if cwd_p.exists():
        return cwd_p
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        target_name = Path(adapter).name
        parent_name = Path(adapter).parent.name
        for candidate in kaggle_input.rglob("adapter_config.json"):
            folder = candidate.parent
            if folder.name == target_name or (parent_name and parent_name in str(folder)):
                return folder
            if target_name in str(folder):
                return folder
    return None


def generate_for_policy(
    cfg,
    policy_name: str,
    dataset: str,
    batch_size: int = 16,
    force: bool = False,
    limit: int | None = None,
):
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(f"Unknown policy: {policy_name}")

    out_dir = repo_path(cfg.get("results_dir", "results")) / "task5_feedback"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"generated_{dataset}_{policy_name}.jsonl"

    if out_file.exists() and not force:
        print(f"[{policy_name}] Found existing responses at {out_file}, skipping generation.")
        return read_jsonl(out_file)

    adapter = specs[policy_name]
    resolved_adapter = None
    if adapter is not None:
        resolved_adapter = find_adapter_path(adapter)
        if resolved_adapter is None:
            raise FileNotFoundError(
                f"Adapter for policy '{policy_name}' not found at '{adapter}'. "
                f"Ensure assets are downloaded via 'python -m scripts.download_assets'."
            )
        print(f"[{policy_name}] Loading policy with adapter: {resolved_adapter}")
    else:
        print(f"[{policy_name}] Loading untouched base policy (SFT baseline)...")

    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=str(resolved_adapter) if resolved_adapter else None, trainable=False)

    rows = read_jsonl(dataset_path(cfg, dataset))
    if limit is not None and limit > 0:
        rows = rows[:limit]

    max_prompt_len = int(cfg.get("max_prompt_length", 256))
    max_new_toks = int(cfg.get("math_max_new_tokens", 512))

    records = []
    print(f"[{policy_name}] Generating math responses for {len(rows)} prompts (batch_size={batch_size})...")

    for start in tqdm(range(0, len(rows), batch_size), desc=f"Generating [{policy_name}]"):
        chunk = rows[start : start + batch_size]
        prompts = [prompt_messages(r) for r in chunk]
        gen = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=max_prompt_len,
            max_new_tokens=max_new_toks,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        for row, response, n_tok in zip(chunk, gen["responses"], gen["response_lengths"]):
            gold_final = str(row.get("gold_final", ""))
            pred_final = extract_designated_final(response)
            is_correct = bool(exact_reward(response, gold_final) == 1.0)
            records.append({
                "source_index": row.get("source_index", len(records)),
                "question": str(row.get("question", row.get("prompt", ""))),
                "gold_final": gold_final,
                "policy": policy_name,
                "dataset": dataset,
                "response": response,
                "pred_final": pred_final,
                "is_correct": is_correct,
                "format_compliant": bool(pred_final is not None),
                "response_tokens": int(n_tok),
            })

    write_jsonl(out_file, records)
    print(f"[{policy_name}] Saved {len(records)} responses to {out_file}\n")
    clear_gpu(model)
    return records


def evaluate_dataset(
    config_path: str = "configs/feedback.yaml",
    dataset: str = "gsm",
    batch_size: int = 16,
    force: bool = False,
    limit: int | None = None,
):
    cfg = load_yaml(config_path)
    out_dir = repo_path(cfg.get("results_dir", "results")) / "task5_feedback"
    out_dir.mkdir(parents=True, exist_ok=True)

    policies = ["sft", "rlvr", "rlaif"]
    records_by_policy = {}

    print("=" * 80)
    print(f"TASK 5: MATH EVALUATION ON DATASET: {dataset.upper()}")
    print("=" * 80)

    # 1. Generate/Load Responses for all 3 policies
    for p in policies:
        records_by_policy[p] = generate_for_policy(
            cfg, p, dataset, batch_size=batch_size, force=force, limit=limit
        )

    n_examples = len(records_by_policy["sft"])
    print(f"\nEvaluating {n_examples} examples across policies: {policies}")

    # 2. Compute Single-Policy Diagnostics (Accuracy, Format Compliance, Length)
    policy_metrics = {}
    for p in policies:
        recs = records_by_policy[p]
        acc = float(np.mean([r["is_correct"] for r in recs]))
        fmt = float(np.mean([r["format_compliant"] for r in recs]))
        lens = [r["response_tokens"] for r in recs]
        policy_metrics[p] = {
            "accuracy": round(acc, 4),
            "format_compliance": round(fmt, 4),
            "mean_length": round(float(np.mean(lens)), 2),
            "std_length": round(float(np.std(lens)), 2),
            "num_examples": len(recs),
        }

    # 3. Pairwise AI Judge Evaluation
    judge_cache_file = out_dir / "ai_judge_cache.json"
    print("\nLoading Pairwise AI Judge for pairwise comparisons...")
    judge = PairwiseAIJudge(cfg, cache_path=judge_cache_file)

    pairs_to_evaluate = [
        ("rlvr", "sft"),
        ("rlaif", "sft"),
        ("rlvr", "rlaif"),
    ]

    pairwise_results = {}

    for name_a, name_b in pairs_to_evaluate:
        recs_a = records_by_policy[name_a]
        recs_b = records_by_policy[name_b]

        wins_a = 0
        wins_b = 0
        ties = 0

        # Verifier-judge agreement tracking
        decisive_agreements = 0
        decisive_comparisons = 0
        full_agreements = 0

        for ra, rb in tqdm(zip(recs_a, recs_b), total=n_examples, desc=f"Pairwise: {name_a.upper()} vs {name_b.upper()}"):
            problem = ra["question"]
            resp_a = ra["response"]
            resp_b = rb["response"]

            pref = judge.compare(problem, resp_a, resp_b)
            if pref == "A":
                wins_a += 1
            elif pref == "B":
                wins_b += 1
            else:
                ties += 1

            # Exact Verifier Preference: +1 if A correct & B wrong; -1 if B correct & A wrong; 0 if equal
            v_pref = int(ra["is_correct"]) - int(rb["is_correct"])
            j_pref = 1 if pref == "A" else (-1 if pref == "B" else 0)

            if v_pref == j_pref:
                full_agreements += 1

            # When either has a preference
            if v_pref != 0 and j_pref != 0:
                decisive_comparisons += 1
                if v_pref == j_pref:
                    decisive_agreements += 1

        win_rate_a = (wins_a + 0.5 * ties) / n_examples
        win_rate_b = (wins_b + 0.5 * ties) / n_examples

        pairwise_results[f"{name_a}_vs_{name_b}"] = {
            "wins_a": wins_a,
            "wins_b": wins_b,
            "ties": ties,
            "win_rate_a": round(win_rate_a, 4),
            "win_rate_b": round(win_rate_b, 4),
            "tie_rate": round(ties / n_examples, 4),
            "decisive_agreement_rate": round(decisive_agreements / max(decisive_comparisons, 1), 4),
            "full_agreement_rate": round(full_agreements / n_examples, 4),
            "decisive_comparisons": decisive_comparisons,
        }

    clear_gpu(judge.model)

    # 4. Assemble Summary Table
    summary_rows = []
    for p in policies:
        pm = policy_metrics[p]
        row = {
            "Policy": p.upper(),
            "Dataset": dataset.upper(),
            "Accuracy (%)": round(pm["accuracy"] * 100, 2),
            "Format Compliance (%)": round(pm["format_compliance"] * 100, 2),
            "Mean Tokens": pm["mean_length"],
        }
        if p == "sft":
            row["Win Rate vs SFT (%)"] = 50.0
        elif p == "rlvr":
            row["Win Rate vs SFT (%)"] = round(pairwise_results["rlvr_vs_sft"]["win_rate_a"] * 100, 2)
        elif p == "rlaif":
            row["Win Rate vs SFT (%)"] = round(pairwise_results["rlaif_vs_sft"]["win_rate_a"] * 100, 2)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 80)
    print(f"EVALUATION RESULTS SUMMARY ({dataset.upper()}):")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    print("\nPAIRWISE COMPARISONS:")
    for k, v in pairwise_results.items():
        print(f"  {k}: Win Rate A={v['win_rate_a']*100:.1f}%, Win Rate B={v['win_rate_b']*100:.1f}%, Tie={v['tie_rate']*100:.1f}% | Decisive Agreement={v['decisive_agreement_rate']*100:.1f}%")

    # Save Results
    out_csv = out_dir / f"{dataset}_evaluation_summary.csv"
    out_json = out_dir / f"{dataset}_evaluation_summary.json"
    summary_df.to_csv(out_csv, index=False)

    full_output = {
        "dataset": dataset,
        "policies": policy_metrics,
        "pairwise": pairwise_results,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    print(f"\nWrote {out_csv} and {out_json}\n")
    return full_output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit on number of examples for rapid debugging")
    args = ap.parse_args()

    evaluate_dataset(
        config_path=args.config,
        dataset=args.dataset,
        batch_size=args.batch_size,
        force=args.force,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
