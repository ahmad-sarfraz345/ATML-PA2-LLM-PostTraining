from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from common.data import load_yaml, repo_path, write_jsonl
from common.generation import batch_generate
from common.models import clear_gpu, load_policy, load_tokenizer


def policy_specs(cfg):
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    return pd.read_csv(repo_path(cfg["paths"]["xstest"]))


def find_adapter_path(adapter: str | None) -> Path | None:
    if not adapter:
        return None
    p = repo_path(adapter)
    if p.exists():
        return p

    cwd_p = Path(adapter)
    if cwd_p.exists():
        return cwd_p

    # Check under /kaggle/input if running in Kaggle
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        parent_name = Path(adapter).parent.name
        policy_tag = parent_name.split("_")[-1] if "_" in parent_name else parent_name
        for candidate in kaggle_input.rglob("adapter_config.json"):
            folder = candidate.parent
            if parent_name and parent_name in str(folder).lower():
                return folder
            if policy_tag and policy_tag in str(folder).lower():
                return folder

    return None


def generate_for_policy(cfg, policy_name: str, batch_size: int = 16, force: bool = False, allow_missing: bool = False):
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(f"Unknown policy: {policy_name}")

    out_dir = repo_path(cfg.get("results_dir", "results")) / "task4_safety"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"generated_{policy_name}.jsonl"

    if out_file.exists() and not force:
        print(f"[{policy_name}] Found existing responses at {out_file}, skipping generation.")
        return out_file

    adapter = specs[policy_name]
    resolved_adapter = None
    if adapter is not None:
        resolved_adapter = find_adapter_path(adapter)
        if resolved_adapter is None:
            msg = (
                f"Adapter path for policy '{policy_name}' not found at '{adapter}'. "
                f"Ensure the standard training run from Task 1/2/3 is completed and outputs are attached or present in outputs/."
            )
            if allow_missing:
                print(f"[{policy_name}] [SKIP] {msg}\n")
                return None
            else:
                raise FileNotFoundError(msg)
        print(f"[{policy_name}] Loading policy with adapter: {resolved_adapter}")
    else:
        print(f"[{policy_name}] Loading untouched base policy (SFT baseline)...")

    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=str(resolved_adapter) if resolved_adapter else None, trainable=False)
    df = load_xstest(cfg)

    max_prompt_len = int(cfg.get("max_prompt_length", 256))
    max_new_toks = int(cfg.get("safety_max_new_tokens", 256))

    records = []
    print(f"[{policy_name}] Generating deterministic responses for {len(df)} XSTest prompts (batch_size={batch_size})...")

    for start in tqdm(range(0, len(df), batch_size), desc=f"Generating [{policy_name}]"):
        chunk = df.iloc[start : start + batch_size]
        prompts = [[{"role": "user", "content": str(x)}] for x in chunk["prompt"].tolist()]
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
        for (_, row), response, n_tok in zip(chunk.iterrows(), gen["responses"], gen["response_lengths"]):
            records.append({
                "xstest_id": int(row["xstest_id"]),
                "policy": policy_name,
                "prompt": str(row["prompt"]),
                "benchmark_class": str(row["benchmark_class"]),
                "type": str(row["type"]),
                "response": response,
                "response_tokens": int(n_tok),
            })

    write_jsonl(out_file, records)
    print(f"[{policy_name}] Saved {len(records)} responses to {out_file}\n")
    clear_gpu(model)
    return out_file


def generate_all_policies(cfg, batch_size: int = 16, force: bool = False, specific_policy: str | None = None, allow_missing: bool = False):
    targets = [specific_policy] if specific_policy and specific_policy != "all" else ["sft", "dpo", "ppo", "grpo"]
    out_files = {}
    for p in targets:
        f = generate_for_policy(cfg, p, batch_size=batch_size, force=force, allow_missing=allow_missing)
        if f is not None:
            out_files[p] = f
    return out_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policy", choices=["all", "sft", "dpo", "ppo", "grpo"], default="all")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--allow-missing", action="store_true", help="Skip policies whose adapters are not found instead of raising an error")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    print("XSTest prompts count:", len(load_xstest(cfg)))
    print("Target policies:", list(policy_specs(cfg)) if args.policy == "all" else [args.policy])
    generate_all_policies(
        cfg,
        batch_size=args.batch_size,
        force=args.force,
        specific_policy=args.policy,
        allow_missing=args.allow_missing,
    )


if __name__ == "__main__":
    main()
