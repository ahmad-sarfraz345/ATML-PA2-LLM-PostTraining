from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import set_seed
from common.models import (
    clear_gpu,
    load_policy,
    load_tokenizer,
    reference_mode,
    trainable_parameters,
)
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            try:
                c_ids, c_mask = encode_prompt_response(tokenizer, prompt, yc, max_length)
                r_ids, r_mask = encode_prompt_response(tokenizer, prompt, yr, max_length)
                chosen.append((c_ids, c_mask))
                rejected.append((r_ids, r_mask))
            except ValueError:
                continue
        if not chosen:
            return None, None
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def run_training(
    config_path: str,
    run_name: str,
    dataset_path: str | None = None,
    output_path: str | None = None,
    beta: float | None = None,
    max_examples: int | None = None,
):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    model = bundle["model"]
    loader = bundle["loader"]
    optimizer = bundle["optimizer"]
    tokenizer = bundle["tokenizer"]
    beta_val = bundle["beta"]

    output = repo_path(output_path or cfg["standard_output"])
    output.mkdir(parents=True, exist_ok=True)
    results_dir = repo_path(cfg.get("results_dir", "results/task1_dpo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    epochs = int(cfg.get("epochs", 1))
    grad_accum_steps = int(cfg.get("grad_accum_steps", 8))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    device = next(model.parameters()).device

    print(f"[{run_name}] Starting DPO training:")
    print(f"  Dataset: {dataset_path or cfg['paths']['dpo_standard_train']}")
    print(f"  Examples: {len(bundle['rows'])}, Batch size: {cfg['batch_size']}, Grad accum: {grad_accum_steps}")
    print(f"  Beta: {beta_val}, Epochs: {epochs}, LR: {cfg['learning_rate']}")
    print(f"  Output path: {output}")

    history = []
    global_step = 0
    running_loss = 0.0
    running_acc = 0.0
    step_in_accum = 0

    optimizer.zero_grad()
    model.train()

    total_batches = len(loader) * epochs
    pbar = tqdm(total=total_batches, desc=f"DPO [{run_name}]")

    for epoch in range(epochs):
        for batch_idx, (chosen_batch, rejected_batch) in enumerate(loader):
            if chosen_batch is None or rejected_batch is None:
                pbar.update(1)
                continue

            chosen_batch = {k: v.to(device) for k, v in chosen_batch.items()}
            rejected_batch = {k: v.to(device) for k, v in rejected_batch.items()}

            # Reference log-probs with disabled LoRA adapter
            with torch.no_grad():
                with reference_mode(model):
                    ref_chosen_logp, _, _ = response_sequence_logprobs(model, chosen_batch)
                    ref_rejected_logp, _, _ = response_sequence_logprobs(model, rejected_batch)

            # Policy log-probs with active trainable LoRA adapter
            policy_chosen_logp, _, _ = response_sequence_logprobs(model, chosen_batch)
            policy_rejected_logp, _, _ = response_sequence_logprobs(model, rejected_batch)

            loss, dpo_diag = dpo_loss(
                policy_chosen_logp=policy_chosen_logp,
                policy_rejected_logp=policy_rejected_logp,
                ref_chosen_logp=ref_chosen_logp,
                ref_rejected_logp=ref_rejected_logp,
                beta=beta_val,
            )

            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()

            running_loss += loss.item()
            running_acc += dpo_diag["preference_accuracy"].item()
            step_in_accum += 1

            if step_in_accum % grad_accum_steps == 0 or (batch_idx + 1) == len(loader):
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(model), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                avg_loss = running_loss / step_in_accum
                avg_acc = running_acc / step_in_accum
                log_entry = {
                    "step": global_step,
                    "epoch": epoch,
                    "batch": batch_idx,
                    "loss": round(avg_loss, 4),
                    "preference_accuracy": round(avg_acc, 4),
                    "policy_margin": round(float(dpo_diag["policy_margin_mean"].item()), 4),
                    "logit_mean": round(float(dpo_diag["logit_mean"].item()), 4),
                    "grad_norm": round(float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm), 4),
                }
                history.append(log_entry)
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{avg_acc:.3f}"})
                global_step += 1
                running_loss = 0.0
                running_acc = 0.0
                step_in_accum = 0

            pbar.update(1)

    pbar.close()

    print(f"[{run_name}] Saving trained LoRA adapter to {output}...")
    model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))

    log_file = results_dir / f"{run_name}_train_log.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[{run_name}] Training log saved to {log_file}")

    clear_gpu(model, optimizer)
    return str(output)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()
