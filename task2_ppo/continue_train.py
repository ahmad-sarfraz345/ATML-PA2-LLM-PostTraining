from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import tqdm

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import (
    batch_generate,
    response_token_logprobs,
    score_reward_pairs,
)
from common.logging_utils import set_seed
from common.metrics import masked_mean, sampled_kl, sample_entropy
from common.models import (
    clear_gpu,
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    reference_mode,
    token_values,
    trainable_parameters,
    value_parameter_groups,
)
from task2_ppo.ppo import (
    compute_gae,
    normalize_advantages,
    ppo_policy_loss,
    shaped_rewards,
    value_mse_loss,
)


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "lora_head"),
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def run_ppo(
    config_path: str,
    output: str | None = None,
    updates: int | None = None,
    clip_epsilon: float | None = None,
    kl_beta: float | None = None,
    run_name: str = "standard",
):
    start_time = time.time()
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]

    num_updates = int(updates if updates is not None else cfg.get("updates", 20))
    clip_eps = float(clip_epsilon if clip_epsilon is not None else cfg.get("clip_epsilon", 0.20))
    beta_kl = float(kl_beta if kl_beta is not None else cfg.get("kl_beta", 0.10))

    policy = bundle["policy"]
    value_model = bundle["value_model"]
    reward_model = bundle["reward_model"]
    reward_tokenizer = bundle["reward_tokenizer"]
    tokenizer = bundle["tokenizer"]
    prompt_rows = bundle["prompt_rows"]
    policy_optimizer = bundle["policy_optimizer"]
    value_optimizer = bundle["value_optimizer"]

    out_path = repo_path(output or cfg["output"])
    out_path.mkdir(parents=True, exist_ok=True)
    results_dir = repo_path(cfg.get("results_dir", "results/task2_ppo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    max_prompt_len = int(cfg.get("max_prompt_length", 256))
    max_resp_len = int(cfg.get("max_response_length", 512))
    ppo_epochs = int(cfg.get("ppo_epochs", 2))
    gamma = float(cfg.get("gamma", 1.0))
    gae_lambda = float(cfg.get("gae_lambda", 0.95))
    value_coef = float(cfg.get("value_coef", 0.50))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    missing_eos_pen = float(cfg.get("missing_eos_penalty", 1.0))

    device = next(policy.parameters()).device

    print(f"[{run_name}] Starting PPO continuation:")
    print(f"  Updates: {num_updates}, Clip epsilon: {clip_eps}, KL beta: {beta_kl}")
    print(f"  PPO Epochs/update: {ppo_epochs}, Max response length: {max_resp_len}")
    print(f"  Output path: {out_path}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    history = []
    pbar = tqdm(total=num_updates, desc=f"PPO [{run_name}]")

    for update_idx in range(num_updates):
        # 1. Sample prompt
        prompt_row = prompt_rows[update_idx % len(prompt_rows)]
        prompt_msgs = prompt_messages(prompt_row)

        # 2. On-policy Rollout Generation
        gen = batch_generate(
            policy,
            tokenizer,
            [prompt_msgs],
            max_prompt_length=max_prompt_len,
            max_new_tokens=max_resp_len,
            temperature=float(cfg.get("generation", {}).get("temperature", 0.7)),
            top_p=float(cfg.get("generation", {}).get("top_p", 0.9)),
            do_sample=True,
        )

        resp_text = gen["responses"][0]
        resp_len = int(gen["response_lengths"][0])
        has_eos = bool(gen["terminated_with_eos"][0])

        if resp_len == 0:
            # Handle edge case of empty response
            pbar.update(1)
            continue

        # 3. Terminal learned reward scoring
        with torch.no_grad():
            raw_rewards = score_reward_pairs(
                reward_model,
                reward_tokenizer,
                [prompt_msgs],
                [resp_text],
                max_length=int(cfg.get("reward_max_length", 1280)),
            )

        raw_reward_val = float(raw_rewards[0].item())
        effective_reward = raw_reward_val if has_eos else (raw_reward_val - missing_eos_pen)
        task_reward_tensor = torch.tensor([effective_reward], device=device, dtype=torch.float32)

        # 4. Old-policy token log-probabilities (eval mode to avoid LoRA dropout noise)
        policy.eval()
        value_model.eval()

        with torch.no_grad():
            old_logp, _ = response_token_logprobs(
                policy,
                gen["sequences"],
                gen["attention_mask"],
                gen["prompt_width"],
                gen["response_ids"],
            )

        # 5. Reference token log-probabilities
        with torch.no_grad():
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(
                    policy,
                    gen["sequences"],
                    gen["attention_mask"],
                    gen["prompt_width"],
                    gen["response_ids"],
                )

        # 6. Value function predictions
        with torch.no_grad():
            full_vals = token_values(value_model, gen["sequences"], gen["attention_mask"])
            p_w = gen["prompt_width"]
            r_tokens = gen["response_ids"].shape[1]
            old_values = full_vals[:, p_w - 1 : p_w - 1 + r_tokens]

        # 7. KL-shaped rewards
        rewards = shaped_rewards(
            task_reward=task_reward_tensor,
            policy_logp=old_logp,
            ref_logp=ref_logp,
            response_mask=gen["response_mask"],
            beta_kl=beta_kl,
        )

        # 8. Generalized Advantage Estimation & Returns
        advantages, returns = compute_gae(
            rewards,
            old_values,
            gen["response_mask"],
            gamma=gamma,
            lam=gae_lambda,
        )
        norm_advantages = normalize_advantages(advantages, gen["response_mask"])

        # 9. PPO Optimization Epochs
        epoch_pol_loss = 0.0
        epoch_val_loss = 0.0
        epoch_clip_frac = 0.0
        epoch_p_norm = 0.0
        latest_new_logp = None

        policy.train()
        value_model.train()

        for _ in range(ppo_epochs):
            # Policy forward pass
            new_logp, _ = response_token_logprobs(
                policy,
                gen["sequences"],
                gen["attention_mask"],
                gen["prompt_width"],
                gen["response_ids"],
            )
            latest_new_logp = new_logp.detach()

            pol_loss, _, clip_frac = ppo_policy_loss(
                new_logp,
                old_logp,
                norm_advantages,
                gen["response_mask"],
                eps=clip_eps,
            )

            policy_optimizer.zero_grad()
            p_norm_val = 0.0
            if not torch.isnan(pol_loss) and not torch.isinf(pol_loss):
                pol_loss.backward()
                p_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_grad_norm)
                p_norm_val = float(p_norm.item() if hasattr(p_norm, "item") else p_norm)
                if not (torch.isnan(p_norm) or torch.isinf(p_norm)):
                    policy_optimizer.step()
                else:
                    policy_optimizer.zero_grad()
            else:
                policy_optimizer.zero_grad()

            # Value model forward pass
            new_full_vals = token_values(value_model, gen["sequences"], gen["attention_mask"])
            new_vals = new_full_vals[:, p_w - 1 : p_w - 1 + r_tokens]
            val_loss = value_mse_loss(new_vals, returns, gen["response_mask"])

            value_optimizer.zero_grad()
            v_norm_val = 0.0
            if not torch.isnan(val_loss) and not torch.isinf(val_loss):
                (value_coef * val_loss).backward()
                v_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(value_model), max_grad_norm)
                v_norm_val = float(v_norm.item() if hasattr(v_norm, "item") else v_norm)
                if not (torch.isnan(v_norm) or torch.isinf(v_norm)):
                    value_optimizer.step()
                else:
                    value_optimizer.zero_grad()
            else:
                value_optimizer.zero_grad()

            epoch_pol_loss += float(pol_loss.item()) if not torch.isnan(pol_loss) else 0.0
            epoch_val_loss += float(val_loss.item()) if not torch.isnan(val_loss) else 0.0
            epoch_clip_frac += float(clip_frac.item()) if not torch.isnan(clip_frac) else 0.0
            epoch_p_norm += p_norm_val

        avg_pol_loss = epoch_pol_loss / ppo_epochs
        avg_val_loss = epoch_val_loss / ppo_epochs
        avg_clip_frac = epoch_clip_frac / ppo_epochs
        avg_p_norm = epoch_p_norm / ppo_epochs

        # Compute diagnostics over response tokens
        with torch.no_grad():
            eval_logp = latest_new_logp if latest_new_logp is not None else old_logp
            cur_kl = float(sampled_kl(eval_logp, ref_logp, gen["response_mask"]).item())
            cur_entropy = float(sample_entropy(eval_logp, gen["response_mask"]).item())

        elapsed = time.time() - start_time
        peak_vram = round(torch.cuda.max_memory_allocated() / (1024**3), 2) if torch.cuda.is_available() else 0.0

        step_record = {
            "update": update_idx + 1,
            "reward": round(raw_reward_val, 4),
            "effective_reward": round(effective_reward, 4),
            "kl": round(cur_kl, 4),
            "policy_loss": round(avg_pol_loss, 4),
            "value_loss": round(avg_val_loss, 4),
            "entropy": round(cur_entropy, 4),
            "clip_fraction": round(avg_clip_frac, 4),
            "gradient_norm": round(avg_p_norm, 4),
            "response_length": resp_len,
            "has_eos": has_eos,
            "wall_clock_time": round(elapsed, 1),
            "peak_vram_gib": peak_vram,
        }
        history.append(step_record)
        log_path = results_dir / f"{run_name}_train_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        pbar.set_postfix({
            "R": f"{raw_reward_val:.2f}",
            "KL": f"{cur_kl:.3f}",
            "P_loss": f"{avg_pol_loss:.3f}",
            "Len": resp_len,
        })
        pbar.update(1)

    pbar.close()
    total_time = time.time() - start_time

    # Save trained policy adapter
    print(f"\n[{run_name}] Saving trained PPO adapter to {out_path}...")
    policy.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))

    # Save training trajectory log
    log_path = results_dir / f"{run_name}_train_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    peak_vram = round(torch.cuda.max_memory_allocated() / (1024**3), 2) if torch.cuda.is_available() else 0.0

    print(f"[{run_name}] PPO continuation finished:")
    print(f"  Wall-clock time: {total_time:.1f}s ({total_time / 60:.2f} min)")
    print(f"  Peak VRAM: {peak_vram} GiB")
    print(f"  Training log: {log_path}\n")

    clear_gpu(policy, value_model, reward_model)
    return {
        "output_path": str(out_path),
        "log_path": str(log_path),
        "wall_clock_time": total_time,
        "peak_vram_gib": peak_vram,
        "history": history,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(
        config_path=args.config,
        output=args.output,
        updates=args.updates,
        clip_epsilon=args.clip_epsilon,
        kl_beta=args.kl_beta,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
