from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import batch_generate, response_sequence_logprobs
from common.metrics import (
    parse_word_limit,
    word_count,
    word_limit_compliance,
)
from common.models import (
    clear_gpu,
    load_policy,
    load_tokenizer,
    reference_mode,
)
from task1_dpo.dpo import dpo_loss
from task1_dpo.evaluate import make_eval_collate
from task1_dpo.train import run_training


def get_stratum_label(row: dict, tokenizer=None) -> str:
    # 1. Exact field from course asset
    if "length_stratum" in row:
        val = str(row["length_stratum"]).lower().strip()
        if "preferred" in val:
            return "preferred_longer"
        if "rejected" in val:
            return "rejected_longer"
        if "match" in val:
            return "length_matched"
        return val

    # 2. Other possible metadata keys
    for key in ["stratum", "strata", "stratum_type", "type"]:
        if key in row:
            val = str(row[key]).lower().strip()
            if "preferred" in val or "chosen" in val:
                return "preferred_longer"
            if "rejected" in val:
                return "rejected_longer"
            if "match" in val:
                return "length_matched"

    # 3. Fallback to token length difference
    if tokenizer is not None:
        yc, yr = preference_responses(row)
        prompt = prompt_messages_from_preference(row)
        c_ids, _ = encode_prompt_response(tokenizer, prompt, yc, max_length=1024)
        r_ids, _ = encode_prompt_response(tokenizer, prompt, yr, max_length=1024)
        p_len = len(tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True))
        len_c = len(c_ids) - p_len
        len_r = len(r_ids) - p_len

        if len_c > len_r * 1.15 and (len_c - len_r) >= 15:
            return "preferred_longer"
        elif len_r > len_c * 1.15 and (len_r - len_c) >= 15:
            return "rejected_longer"
        else:
            return "length_matched"
    return "length_matched"


def evaluate_stratified_preference(model, tokenizer, rows, cfg):
    device = next(model.parameters()).device
    max_seq_len = int(cfg.get("max_sequence_length", 768))
    loader = DataLoader(
        rows,
        batch_size=int(cfg.get("batch_size", 2)) * 2,
        shuffle=False,
        collate_fn=make_eval_collate(tokenizer, max_seq_len),
    )

    strata_results = defaultdict(lambda: {"margins": [], "accuracies": []})
    all_margins = []

    with torch.no_grad():
        for chosen_batch, rejected_batch, valid_indices in loader:
            if chosen_batch is None:
                continue
            chosen_batch = {k: v.to(device) for k, v in chosen_batch.items()}
            rejected_batch = {k: v.to(device) for k, v in rejected_batch.items()}

            with reference_mode(model):
                ref_chosen_logp, _, _ = response_sequence_logprobs(model, chosen_batch)
                ref_rejected_logp, _, _ = response_sequence_logprobs(model, rejected_batch)

            pol_chosen_logp, _, _ = response_sequence_logprobs(model, chosen_batch)
            pol_rejected_logp, _, _ = response_sequence_logprobs(model, rejected_batch)

            m = (pol_chosen_logp - pol_rejected_logp) - (ref_chosen_logp - ref_rejected_logp)
            m_list = m.cpu().tolist()

            for idx, m_val in zip(valid_indices, m_list):
                row = rows[idx]
                stratum = get_stratum_label(row, tokenizer)
                is_correct = 1.0 if m_val > 0 else 0.0
                strata_results[stratum]["margins"].append(m_val)
                strata_results[stratum]["accuracies"].append(is_correct)
                all_margins.append(m_val)

    summary = {}
    for st, data in strata_results.items():
        summary[st] = {
            "count": len(data["accuracies"]),
            "accuracy": round(float(np.mean(data["accuracies"])), 4),
            "mean_margin": round(float(np.mean(data["margins"])), 4),
        }
    summary["overall"] = {
        "count": len(all_margins),
        "accuracy": round(float(np.mean([1.0 if m > 0 else 0.0 for m in all_margins])), 4),
        "mean_margin": round(float(np.mean(all_margins)), 4),
    }
    return summary


def evaluate_word_limits(model, tokenizer, prompt_rows, cfg=None, max_tokens=256):
    prompts = [prompt_messages(r) for r in prompt_rows]
    gen_cfg = (cfg or {}).get("generation", {})
    temp = float(gen_cfg.get("temperature", 0.7))
    top_p = float(gen_cfg.get("top_p", 0.9))
    do_sample = bool(gen_cfg.get("do_sample", True))
    gen_out = batch_generate(
        model,
        tokenizer,
        prompts,
        max_prompt_length=256,
        max_new_tokens=max_tokens,
        temperature=temp,
        top_p=top_p,
        do_sample=do_sample,
    )

    results = []
    for r, response, n_tok in zip(prompt_rows, gen_out["responses"], gen_out["response_lengths"]):
        prompt_text = str(r.get("prompt", r.get("messages", [{}])[-1].get("content", "")))
        pid = r.get("prompt_id", "unknown")
        limit = parse_word_limit(prompt_text)
        w_cnt = word_count(response)
        comp = word_limit_compliance(prompt_text, response)
        results.append({
            "prompt_id": pid,
            "word_limit": limit,
            "actual_words": w_cnt,
            "response_tokens": int(n_tok),
            "compliant": bool(comp == 1.0) if comp is not None else None,
            "response": response,
        })

    comp_rates = [1.0 if r["compliant"] else 0.0 for r in results if r["compliant"] is not None]
    overall_comp = float(np.mean(comp_rates)) if comp_rates else 0.0
    mean_words = float(np.mean([r["actual_words"] for r in results])) if results else 0.0
    mean_tokens = float(np.mean([r["response_tokens"] for r in results])) if results else 0.0

    return {
        "compliance_rate": round(overall_comp, 4),
        "mean_words": round(mean_words, 2),
        "mean_tokens": round(mean_tokens, 2),
        "details": results,
    }


def run_length_analysis(config_path: str, force_train: bool = False):
    cfg = load_yaml(config_path)
    results_dir = repo_path(cfg.get("results_dir", "results/task1_dpo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    std_adapter_path = repo_path(cfg["standard_output"])
    len_adapter_path = repo_path(cfg["length_output"])

    print("=" * 60)
    print("TASK 1: DPO LENGTH-CONFOUNDING STUDY")
    print("=" * 60)

    # 1. Train length-balanced model if not yet trained
    if not (len_adapter_path / "adapter_config.json").exists() or force_train:
        print(f"\n>>> Training length-balanced DPO model from {cfg['paths']['dpo_length_train']}")
        run_training(
            config_path=config_path,
            run_name="length_balanced",
            dataset_path=cfg["paths"]["dpo_length_train"],
            output_path=cfg["length_output"],
        )
    else:
        print(f"\n>>> Found existing length-balanced adapter at {len_adapter_path}")

    tokenizer = load_tokenizer(cfg["base_model"])
    stratified_rows = read_jsonl(cfg["paths"]["dpo_length_eval"])
    word_limit_rows = read_jsonl(cfg["paths"]["word_limit_prompts"])

    # 2. Evaluate Standard Model
    print(f"\n>>> Evaluating Standard DPO model on stratified eval ({len(stratified_rows)} rows)")
    std_policy = load_policy(cfg, adapter_path=cfg["standard_output"], trainable=False)
    std_stratified = evaluate_stratified_preference(std_policy, tokenizer, stratified_rows, cfg)
    print(">>> Evaluating Standard DPO model on word-limit prompts")
    std_word_limits = evaluate_word_limits(std_policy, tokenizer, word_limit_rows, cfg=cfg)
    clear_gpu(std_policy)

    # 3. Evaluate Length-Balanced Model
    print(f"\n>>> Evaluating Length-Balanced DPO model on stratified eval ({len(stratified_rows)} rows)")
    len_policy = load_policy(cfg, adapter_path=cfg["length_output"], trainable=False)
    len_stratified = evaluate_stratified_preference(len_policy, tokenizer, stratified_rows, cfg)
    print(">>> Evaluating Length-Balanced DPO model on word-limit prompts")
    len_word_limits = evaluate_word_limits(len_policy, tokenizer, word_limit_rows, cfg=cfg)
    clear_gpu(len_policy)

    # 4. Compare Stratified Accuracy
    strata_order = ["preferred_longer", "length_matched", "rejected_longer", "overall"]
    table_rows = []
    for st in strata_order:
        s_data = std_stratified.get(st, {"count": 0, "accuracy": 0.0, "mean_margin": 0.0})
        l_data = len_stratified.get(st, {"count": 0, "accuracy": 0.0, "mean_margin": 0.0})
        table_rows.append({
            "Stratum": st,
            "Count": s_data.get("count", 0),
            "Standard Acc (%)": round(s_data.get("accuracy", 0) * 100, 2),
            "Length-Balanced Acc (%)": round(l_data.get("accuracy", 0) * 100, 2),
            "Standard Margin": s_data.get("mean_margin", 0),
            "Length-Balanced Margin": l_data.get("mean_margin", 0),
        })

    stratified_df = pd.DataFrame(table_rows)

    # 5. Compare Word-Limit Compliance
    word_comp_table = []
    for s_item, l_item in zip(std_word_limits["details"], len_word_limits["details"]):
        word_comp_table.append({
            "Prompt ID": s_item["prompt_id"],
            "Word Limit": s_item["word_limit"],
            "Std Words": s_item["actual_words"],
            "Std Compliant": s_item["compliant"],
            "Balanced Words": l_item["actual_words"],
            "Balanced Compliant": l_item["compliant"],
        })
    word_comp_df = pd.DataFrame(word_comp_table)

    # 6. Save combined results
    combined_summary = {
        "stratified_eval": {
            "standard": std_stratified,
            "length_balanced": len_stratified,
        },
        "word_limit_compliance": {
            "standard": {
                "compliance_rate": std_word_limits["compliance_rate"],
                "mean_words": std_word_limits["mean_words"],
                "mean_tokens": std_word_limits["mean_tokens"],
            },
            "length_balanced": {
                "compliance_rate": len_word_limits["compliance_rate"],
                "mean_words": len_word_limits["mean_words"],
                "mean_tokens": len_word_limits["mean_tokens"],
            },
        },
    }

    out_json = results_dir / "length_confounding_analysis.json"
    out_csv = results_dir / "length_confounding_analysis.csv"
    word_csv = results_dir / "word_limit_compliance.csv"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(combined_summary, f, indent=2)
    stratified_df.to_csv(out_csv, index=False)
    word_comp_df.to_csv(word_csv, index=False)

    print("\n" + "=" * 75)
    print("STRATIFIED PREFERENCE ACCURACY COMPARISON (LENGTH CONFOUNDING)")
    print("=" * 75)
    print(stratified_df.to_string(index=False))
    print("=" * 75)

    print("\n" + "=" * 75)
    print("EXPLICIT WORD-LIMIT COMPLIANCE")
    print("=" * 75)
    print(f"Standard Model Compliance:        {std_word_limits['compliance_rate'] * 100:.1f}% (Mean words: {std_word_limits['mean_words']})")
    print(f"Length-Balanced Model Compliance: {len_word_limits['compliance_rate'] * 100:.1f}% (Mean words: {len_word_limits['mean_words']})")
    print("=" * 75)
    print(word_comp_df.to_string(index=False))
    print("=" * 75)
    print(f"\nSaved analysis JSON: {out_json}")
    print(f"Saved stratified CSV: {out_csv}")
    print(f"Saved word limit CSV: {word_csv}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--force-train", action="store_true")
    args = ap.parse_args()
    run_length_analysis(args.config, args.force_train)


if __name__ == "__main__":
    main()
