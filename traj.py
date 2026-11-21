"""Turn-level trajectory storage.

Each turn keeps the exact token context the sampler fed the policy, the sampled action tokens, their
behaviour log-probs, and the policy version. The trainer scores these stored contexts directly; it never
rebuilds context from text or from full history, which would silently make the data off-policy.
"""
from dataclasses import dataclass, replace

import numpy as np

from env import TOK


@dataclass
class Rollout:
    ctx: list    # per turn t: (B, L_t) int64, exactly what the policy saw
    act: np.ndarray     # (B, H, 2) agent [verb, arg] tokens: the only loss targets
    logp: np.ndarray    # (B, H, 2) behaviour log-probs at sampling time
    mem: np.ndarray     # (B, H) persistent memory before acting
    reward: np.ndarray  # (B, H)
    goal: np.ndarray    # (B,) initial goal index
    version: np.ndarray  # (B, H) policy version that generated each turn (episodes span many versions)
    branch: tuple = (0, 1)  # (b, k): episodes share their first b turns in blocks of k consecutive rows

    @property
    def ret(self):
        return self.reward.sum(1)

    def rows(self, sl):
        """The episodes in slice sl, for minibatch steps. Keep whole groups (and whole forks) together."""
        return replace(self, ctx=[c[sl] for c in self.ctx], act=self.act[sl], logp=self.logp[sl], mem=self.mem[sl],
                       reward=self.reward[sl], goal=self.goal[sl], version=self.version[sl])


def build_context(mem, obs_hist, act_hist, W):
    """Context-management policy: [<mem> m] + the last W-1 full turns + current obs + <agent>."""
    t, B = len(obs_hist) - 1, len(mem)
    parts = [np.stack([np.full(B, TOK["<mem>"]), mem], 1)]
    for j in range(max(0, t - W + 1), t):
        parts += [obs_hist[j], act_hist[j]]
    parts += [obs_hist[t], np.full((B, 1), TOK["<agent>"])]
    return np.concatenate(parts, 1)


def max_context_len(H, W):
    return 6 * min(H, W) + 1  # longest context plus the verb token appended when scoring
