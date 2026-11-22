"""Training objectives on stale data, and drift diagnostics.

logp: (B, H, 2) agent-token log-probs under the current trainer weights (with grad).
beh:  (B, H, 2) stored behaviour log-probs from whichever sampler version produced each turn.
Ratios are always taken against the stored behaviour log-probs; recomputing "old" log-probs with the
trainer's weights would make every ratio 1 and hide staleness.

none / ppo   use per-turn advantages (B, H) from credit.py.
oapl / coro  regress the episode log-ratio l_i = sum_t log(pi/mu) onto episode returns within groups of
             G episodes from the same seed (Ritter et al. 2026; humans& Ampersand blog). No ratios.
"""
import math

import numpy as np
import torch


def weighted_log_ratio(logp, beh, beta):
    """(B, H) per-turn log-ratios times beta; beta is a scalar or (B, H) per-turn weights (one dual
    variable per staleness bucket, so turns from older samplers can be regularised harder)."""
    return beta * (logp - beh).sum(2)


def loss(cfg, logp, beh, adv, ret, beta):
    if cfg.objective == "none":  # treat stale data as on-policy
        return -(adv[..., None] * logp).mean()
    if cfg.objective == "ppo":  # token-level clipped ratio against the behaviour policy
        r, a = (logp - beh).exp(), adv[..., None]
        return -torch.minimum(r * a, r.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
    G = cfg.group_size
    l, r = weighted_log_ratio(logp, beh, beta).sum(1).view(-1, G), ret.view(-1, G)  # l = beta * l_i
    if cfg.objective == "oapl":
        v_hat = cfg.beta1 * (torch.logsumexp(r / cfg.beta1, 1, keepdim=True) - math.log(G))
        return ((l - (r - v_hat)) ** 2).sum(1).mean()
    if cfg.objective == "coro":
        return (((l - l.mean(1, keepdim=True)) - (r - r.mean(1, keepdim=True))) ** 2).sum(1).mean()
    raise ValueError(f"unknown objective {cfg.objective!r}")


def tree_loss(cfg, logp, beh, rew, branch, beta):
    """Branched CORO. G1 = cfg.group_size trunks per seed each fork into k continuations at turn b.
    Level 1 contrasts trunks on their prefix log-ratio against prefix reward + mean continuation return
    (a Monte Carlo value of the branch state). Level 2 contrasts the k continuations of one trunk on
    their suffix only, so no turn is credited for a reward that came before it or for a sibling's luck.
    With b = 0 the prefix is empty and this is CORO with groups of k (times G1)."""
    b, k = branch
    G1 = cfg.group_size
    lt, rt = weighted_log_ratio(logp, beh, beta), rew  # (n*G1*k, H), already times beta
    l_pre, r_pre = (x[:, :b].sum(1).view(-1, G1, k)[:, :, 0] for x in (lt, rt))
    l_suf, r_suf = (x[:, b:].sum(1).view(-1, G1, k) for x in (lt, rt))
    c = lambda x, d: x - x.mean(d, keepdim=True)  # noqa: E731
    lvl1 = ((c(l_pre, 1) - c(r_pre + r_suf.mean(2), 1)) ** 2).sum(1)
    if cfg.tree_no_level1:  # control: sibling contrast only, so prefix turns get no gradient
        lvl1 = torch.zeros_like(lvl1)
    lvl2 = ((c(l_suf, 2) - c(r_suf, 2)) ** 2).sum((1, 2))
    return (lvl1 + lvl2).mean()


def tree_advantages(rew, branch, G1, eps=1e-6, std=True):
    """Branched PPO-clip baseline (VinePPO / Tree-GRPO style) on the same forked data: the per-turn
    advantages that Branched CORO's two levels imply, standardised as in credit.A0 (std=False: centred
    only, as in credit.A0dr and as CORO's targets are). Suffix turns get the suffix return standardised
    among the k siblings. Prefix turns get prefix reward + mean continuation return standardised across
    the G1 trunks, on sibling 0 only, since the k copies of a prefix are one sample (as in tree_loss
    level 1). rew: (n*G1*k, H) numpy. Returns (n*G1*k, H) float32."""
    b, k = branch
    H = rew.shape[1]
    r_pre = rew[:, :b].sum(1).reshape(-1, G1, k)[:, :, 0]
    r_suf = rew[:, b:].sum(1).reshape(-1, G1, k)
    z = lambda x, d: (x - x.mean(d, keepdims=True)) / ((x.std(d, keepdims=True) + eps) if std else 1.0)  # noqa: E731
    adv = np.zeros((len(rew) // (G1 * k), G1, k, H), np.float32)
    adv[..., b:] = z(r_suf, 2)[..., None]
    adv[:, :, 0, :b] = z(r_pre + r_suf.mean(2), 1)[..., None]
    return adv.reshape(rew.shape)


def turn_kl_penalty(logp, beh, lam):
    """sum over turns of lam[b,t] * KL_k3 at that turn, averaged over episodes.

    Unlike a per-turn beta, this leaves the contrastive term's gradient alone: at a turn with no drift
    (logp == beh) its gradient is exactly zero, whatever lam is."""
    lr = logp - beh
    return (lam * (lr.exp() - 1 - lr).mean(2)).sum(1).mean()


def drift_stats(logp, beh, group_size):
    lr = (logp - beh).detach()
    ell = lr.sum((1, 2))
    w = (ell - ell.max()).exp()  # trajectory importance weights, max-normalised for stability
    return {
        "log_ratio_abs": lr.abs().mean().item(),
        "ratio_max": lr.exp().max().item(),
        "kl_k3": (lr.exp() - 1 - lr).mean().item(),  # per-token KL(behaviour || current), k3 estimator
        "ell_std": ell.view(-1, group_size).std(1).mean().item(),  # within-group spread of l_i
        "ess_traj": (w.sum() ** 2 / (len(w) * (w**2).sum())).item(),  # normalised ESS of exp(l_i)
    }
