"""The real async mode: weights swap only at turn boundaries, and each turn carries the version that sampled it."""
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parents[1]))
from async_runner import AsyncConfig, copy_into, sample_batch, train_async  # noqa: E402
from policy import Policy  # noqa: E402
from runner import score_rollout  # noqa: E402
from traj import max_context_len  # noqa: E402


def tiny(**kw):
    return AsyncConfig(**{**dict(H=8, W=8, groups=2, group_size=4, iters=4, eval_every=1000, eval_episodes=8, samplers=2), **kw})


def test_versions_follow_the_weights_that_sampled_each_turn():
    cfg = tiny(objective="tree", group_size=2, branch=2)
    torch.manual_seed(0)
    a, b = Policy(max_context_len(cfg.H, cfg.W)), Policy(max_context_len(cfg.H, cfg.W))
    model = Policy(max_context_len(cfg.H, cfg.W))
    copy_into(model, nn.utils.parameters_to_vector(a.parameters()))
    calls = []

    def pull():  # the trainer publishes version 1 (= b) while the episode is at turn 3
        calls.append(1)
        if len(calls) == 4:
            copy_into(model, nn.utils.parameters_to_vector(b.parameters()).detach())
        return int(len(calls) >= 4)

    ro, branch = sample_batch(cfg, model, pull, np.random.default_rng(0), torch.Generator().manual_seed(0))
    assert len(calls) == cfg.H and len(ro.act) == cfg.groups * cfg.group_size * branch[1]
    assert (ro.version == (np.arange(cfg.H) >= 3)[None]).all()
    with torch.no_grad():
        lp_a, lp_b = score_rollout(a, ro).numpy(), score_rollout(b, ro).numpy()
    old = ro.version == 0
    assert np.abs(lp_a - ro.logp)[old].max() < 1e-5 and np.abs(lp_b - ro.logp)[~old].max() < 1e-5
    assert np.abs(lp_a - ro.logp)[~old].max() > 1e-3  # the swap really changed the policy


def test_async_trains_end_to_end():
    for objective, extra in (("tree", dict(group_size=2, branch=2)), ("coro", dict(drift_budget=0.002))):
        _, metrics = train_async(tiny(objective=objective, **extra), log=False)
        assert len(metrics) == 4 and all(np.isfinite(m["loss"]) for m in metrics)
        assert all(m["staleness"] >= 0 and m["stal_first"] >= m["stal_last"] for m in metrics)
