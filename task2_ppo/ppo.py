from __future__ import annotations

import torch

from common.metrics import masked_mean


def compute_gae(rewards, values, mask, gamma=1.0, lam=0.95):
    """Token-level GAE over response positions.

    rewards, values, mask: [batch, response_steps]. Padding positions must have mask=0.
    The final valid response position bootstraps with zero.
    """
    rewards = rewards.float()
    values = values.float()
    mask = mask.float()
    batch, steps = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(batch, device=rewards.device, dtype=torch.float32)

    for t in reversed(range(steps)):
        current_valid = mask[:, t]
        if t + 1 < steps:
            next_valid = mask[:, t + 1]
            next_value = values[:, t + 1] * next_valid
        else:
            next_valid = torch.zeros_like(current_valid)
            next_value = torch.zeros_like(last_adv)

        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_adv = delta + gamma * lam * next_valid * last_adv
        last_adv = last_adv * current_valid
        advantages[:, t] = last_adv

    returns = advantages + values
    return advantages, returns


def shaped_rewards(task_reward, policy_logp, ref_logp, response_mask, beta_kl):
    """Sampled-action KL shaping plus terminal learned reward."""
    rewards = -float(beta_kl) * (policy_logp.float() - ref_logp.float()) * response_mask.float()
    for b in range(rewards.shape[0]):
        valid = int(response_mask[b].sum().item())
        if valid > 0:
            rewards[b, valid - 1] += float(task_reward[b].item() if hasattr(task_reward[b], "item") else task_reward[b])
    return rewards


def ppo_policy_loss(new_logp, old_logp, advantage, mask, eps=0.2):
    """Return PPO clipped policy loss and diagnostics.

    Validate this starter implementation against the clipped surrogate in the assignment manual.
    """
    log_ratio = (new_logp.float() - old_logp.float()).clamp(-20.0, 20.0)
    ratio = torch.exp(log_ratio)
    surr1 = ratio * advantage.float()
    surr2 = ratio.clamp(1.0 - eps, 1.0 + eps) * advantage.float()

    objective = torch.minimum(surr1, surr2)

    loss = -masked_mean(objective, mask)
    affected = ((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).float()
    clip_fraction = masked_mean(affected, mask)
    return loss, ratio.detach(), clip_fraction.detach()


def value_mse_loss(predicted_values, returns, mask):
    diff = predicted_values.float() - returns.float()
    return masked_mean(diff ** 2, mask)


def normalize_advantages(advantages, mask, eps=1e-6):
    mask_b = mask.bool()
    adv_f = advantages.float()
    valid = adv_f[mask_b]
    if valid.numel() <= 1:
        return torch.where(mask_b, adv_f, torch.zeros_like(adv_f))
    mean = valid.mean()
    std = valid.std(unbiased=False).clamp_min(eps)
    normed = (adv_f - mean) / std
    return torch.where(mask_b, normed, torch.zeros_like(adv_f))
