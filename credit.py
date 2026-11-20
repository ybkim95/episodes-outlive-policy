"""Credit assignment: map a rollout to per-turn advantages (B, H), applied to both tokens of a turn."""
import numpy as np


def outcome_broadcast(ro, cfg, eps=1e-6):
    """A0 (GRPO-style): episode return minus group mean, over group std, copied to every turn."""
    r = ro.ret.reshape(-1, cfg.group_size)
    adv = ((r - r.mean(1, keepdims=True)) / (r.std(1, keepdims=True) + eps)).reshape(-1)
    return np.repeat(adv[:, None], ro.act.shape[1], 1).astype(np.float32)


def outcome_broadcast_dr(ro, cfg):
    """A0 without the std division (Dr. GRPO). On-policy, CORO's gradient points the same way."""
    r = ro.ret.reshape(-1, cfg.group_size)
    adv = (r - r.mean(1, keepdims=True)).reshape(-1)
    return np.repeat(adv[:, None], ro.act.shape[1], 1).astype(np.float32)


CREDIT = {"A0": outcome_broadcast, "A0dr": outcome_broadcast_dr}
