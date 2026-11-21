"""Sanity checks that must pass before any result is trusted (plan section 5)."""
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from env import ARG_IDS, GOAL_IDS, TOK, VERB_IDS, MemoryDialogue  # noqa: E402
from offpolicy import loss  # noqa: E402
from policy import Policy  # noqa: E402
from runner import Config, rollout, score_rollout, staleness, train  # noqa: E402
from traj import build_context, max_context_len  # noqa: E402


def small_cfg(**kw):
    return Config(**{**dict(H=10, W=4, groups=4, group_size=4, iters=6, eval_every=1000, eval_episodes=16), **kw})


def fresh_policy(cfg, seed):
    torch.manual_seed(seed)
    return Policy(max_context_len(cfg.H, cfg.W))


def small_rollout(cfg, seed=0, stal=None, models=None):
    models = models or [fresh_policy(cfg, seed)]
    stal = np.zeros((16, cfg.H), int) if stal is None else stal
    ro = rollout(lambda s: models[s], cfg, np.arange(16) % 8, np.random.default_rng(seed),
                 torch.Generator().manual_seed(seed), stal)
    return models[0], ro


# 1. stored context + behaviour weights reproduce the stored log-probs
def test_logprob_consistency():
    model, ro = small_rollout(small_cfg())
    with torch.no_grad():
        lp = score_rollout(model, ro).numpy()
    assert np.abs(lp - ro.logp).max() < 1e-5


# 1b. with a different sampler version per episode and turn, each turn's stored log-probs match the
#     version it is tagged with (and only that one)
def test_per_turn_versions():
    cfg = small_cfg(U=3, qmax=2)
    stal = staleness(cfg, np.random.default_rng(0), 16)
    assert (np.diff(stal, axis=1) <= 0).all() and len(np.unique(stal)) > 2  # earlier turns are older
    models = [fresh_policy(cfg, s) for s in range(stal.max() + 1)]
    _, ro = small_rollout(cfg, stal=stal, models=models)
    for s, m in enumerate(models):
        with torch.no_grad():
            lp = score_rollout(m, ro).numpy()
        tagged = stal == s
        assert np.abs(lp - ro.logp)[tagged].max() < 1e-5
        assert np.abs(lp - ro.logp)[~tagged].max() > 1e-3


# 2. the controlled-lag machinery at k=0 is exactly synchronous training
def test_lag0_matches_sync():
    for objective in ["none", "coro"]:
        m_sync, _ = train(small_cfg(sync=True, objective=objective), log=False)
        m_lag, _ = train(small_cfg(lag=0, objective=objective), log=False)
        for a, b in zip(m_sync.state_dict().values(), m_lag.state_dict().values()):
            assert torch.equal(a, b)


# 3. ratios are taken against stored behaviour log-probs, so staleness shows up in them
def test_ratio_uses_behaviour_logprobs():
    _, m0 = train(small_cfg(lag=0), log=False)
    assert max(m["log_ratio_abs"] for m in m0) < 1e-5
    _, m3 = train(small_cfg(lag=3, iters=10), log=False)
    assert [m["staleness"] for m in m3] == [min(m["iter"], 3) for m in m3]
    assert max(m["log_ratio_abs"] for m in m3[4:]) > 1e-4


# 4. the loss never predicts user/tool/teammate tokens: their output rows get exactly zero gradient
def test_only_agent_tokens_are_targets():
    model, ro = small_rollout(small_cfg())
    assert np.isin(ro.act[..., 0], VERB_IDS).all() and np.isin(ro.act[..., 1], ARG_IDS).all()
    score_rollout(model, ro).sum().backward()
    non_agent = ~(model.verb_ok | model.arg_ok)
    assert model.head.weight.grad[non_agent].abs().max() == 0


# 5. objective identities used in the write-up
def random_batch(G=8, n=4, H=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    logp = torch.randn(n * G, H, 2, generator=g)
    beh = torch.randn(n * G, H, 2, generator=g)
    ret = torch.rand(n * G, generator=g)
    return logp, beh, ret


def test_coro_ignores_uniform_drift():
    cfg = small_cfg(objective="coro", group_size=8)
    logp, beh, ret = random_batch()
    shifted = logp + 0.3  # every trajectory's log-ratio moves by the same amount
    assert torch.allclose(loss(cfg, logp, beh, None, ret, 0.5), loss(cfg, shifted, beh, None, ret, 0.5))


def test_oapl_is_coro_plus_calibration():
    G, beta = 8, 0.5
    cfg_o, cfg_c = small_cfg(objective="oapl", group_size=G), small_cfg(objective="coro", group_size=G)
    logp, beh, ret = random_batch(G)
    ell, r = (logp - beh).sum((1, 2)).view(-1, G), ret.view(-1, G)
    v_hat = cfg_o.beta1 * (torch.logsumexp(r / cfg_o.beta1, 1) - math.log(G))
    calib = G * (beta * ell.mean(1) - r.mean(1) + v_hat) ** 2
    lhs = loss(cfg_o, logp, beh, None, ret, beta)
    assert torch.allclose(lhs, loss(cfg_c, logp, beh, None, ret, beta) + calib.mean(), rtol=1e-5)


def test_coro_is_all_pairs_rebel():
    G, beta = 8, 0.5
    logp, beh, ret = random_batch(G)
    a = (beta * (logp - beh).sum((1, 2)) - ret).view(-1, G)
    rebel = ((a[:, :, None] - a[:, None, :]) ** 2).sum((1, 2)) / (2 * G)
    coro = loss(small_cfg(objective="coro", group_size=G), logp, beh, None, ret, beta)
    assert torch.allclose(coro, rebel.mean(), rtol=1e-5)


def test_fork_shares_prefix_then_diverges():
    cfg, b, k = small_cfg(), 4, 3
    model = fresh_policy(cfg, 0)
    ro = rollout(lambda s: model, cfg, np.arange(5) % 8, np.random.default_rng(0),
                 torch.Generator().manual_seed(0), np.zeros((5, cfg.H), int), branch=(b, k))
    assert ro.act.shape[0] == 5 * k
    sib = lambda x: x.reshape(5, k, *x.shape[1:])  # noqa: E731
    acts, logps = sib(ro.act), sib(ro.logp)
    assert (acts[:, :, :b] == acts[:, :1, :b]).all() and (logps[:, :, :b] == logps[:, :1, :b]).all()
    assert all((sib(ro.ctx[t]) == sib(ro.ctx[t])[:, :1]).all() for t in range(b + 1))  # turn b's input too
    assert (acts[:, :, b:] != acts[:, :1, b:]).any()


def test_tree_reduces_to_coro_at_root():
    from offpolicy import tree_loss
    G1, k, beta = 4, 3, 0.5
    logp, beh, ret = random_batch(G=G1 * k, n=2)
    rew = torch.zeros(len(ret), logp.shape[1])
    rew[:, -1] = ret
    grads = []
    for f in (lambda lp: tree_loss(small_cfg(group_size=G1), lp, beh, rew, (0, k), beta),
              lambda lp: G1 * loss(small_cfg(objective="coro", group_size=k), lp, beh, None, ret, beta)):
        lp = logp.clone().requires_grad_()
        f(lp).backward()
        grads.append(lp.grad)
    assert torch.allclose(*grads, atol=1e-6)


def test_tree_levels_split_credit():
    # b > 0: level 1 is CORO over trunks (one shared prefix each) on prefix reward + mean continuation
    # return; level 2 is CORO over siblings on their suffix only, so earlier rewards never reach suffix turns
    from offpolicy import tree_loss
    G1, k, b, H, beta = 4, 3, 2, 5, 0.5
    tr = lambda x: x.view(-1, G1, k, *x.shape[1:])  # noqa: E731

    def fork(x):  # siblings share the first b turns, as in a real fork
        x = tr(x.clone())
        x[:, :, 1:, :b] = x[:, :, :1, :b]
        return x.view(-1, *x.shape[3:])

    logp, beh, _ = random_batch(G=G1 * k, n=2, H=H)
    logp, beh, rew = map(fork, (logp, beh, torch.rand(len(logp), H, generator=torch.Generator().manual_seed(1))))
    lp = logp.clone().requires_grad_()
    tree_loss(small_cfg(group_size=G1), lp, beh, rew, (b, k), beta).backward()
    g = tr(lp.grad)
    lp1 = tr(logp)[:, :, 0, :b].clone().requires_grad_()
    r1 = tr(rew)[:, :, 0, :b].sum(-1) + tr(rew[:, b:].sum(1)).mean(2)
    loss(small_cfg(objective="coro", group_size=G1), lp1.reshape(-1, b, 2), tr(beh)[:, :, 0, :b].reshape(-1, b, 2),
         None, r1.reshape(-1), beta).backward()
    assert torch.allclose(g[:, :, 0, :b], lp1.grad, atol=1e-6) and (g[:, :, 1:, :b] == 0).all()
    lp2 = logp[:, b:].clone().requires_grad_()
    (G1 * loss(small_cfg(objective="coro", group_size=k), lp2, beh[:, b:], None, rew[:, b:].sum(1), beta)).backward()
    assert torch.allclose(lp.grad[:, b:], lp2.grad, atol=1e-6)


def test_two_pass_backward_is_exact():
    from offpolicy import tree_loss
    from runner import backward_two_pass
    cfg = small_cfg(objective="coro", group_size=4)
    model = fresh_policy(cfg, 0)
    ro = rollout(lambda s: model, cfg, np.arange(4) % 8, np.random.default_rng(0),
                 torch.Generator().manual_seed(0), np.zeros((4, cfg.H), int), branch=(3, 4))
    beh = torch.as_tensor(ro.logp) - 0.05 * torch.rand(16, cfg.H, 2)  # make log-ratios non-zero
    rew = torch.as_tensor(ro.reward) + torch.rand(16, cfg.H)
    objectives = [lambda lp: loss(cfg, lp, beh, None, rew.sum(1), 0.3),
                  lambda lp: tree_loss(small_cfg(group_size=1), lp, beh, rew, (3, 4), 0.3)]
    for f in objectives:
        model.zero_grad()
        f(score_rollout(model, ro)).backward()
        one = [p.grad.clone() for p in model.parameters()]
        model.zero_grad()
        backward_two_pass(model, ro, f)
        assert all(torch.allclose(a, p.grad, atol=1e-6) for a, p in zip(one, model.parameters()))


# 5b. the per-turn KL penalty (E7) never touches a turn that has not drifted, unlike a per-turn beta (E6)
def test_turn_penalty_spares_fresh_turns():
    from offpolicy import turn_kl_penalty
    logp, beh, _ = random_batch()
    beh[:, -2:] = logp[:, -2:]  # last two turns are fresh: trainer == sampler there
    lp = logp.clone().requires_grad_()
    turn_kl_penalty(lp, beh, torch.full(logp.shape[:2], 5.0)).backward()
    assert lp.grad[:, -2:].abs().max() == 0 and lp.grad[:, :-2].abs().min() > 0
    # per-turn beta at its floor shrinks the contrastive gradient at that turn by the same factor
    cfg = small_cfg(objective="coro", group_size=8)
    w = torch.full(logp.shape[:2], 0.3)
    w[:, -2:] = 1e-3
    lp = logp.clone().requires_grad_()
    loss(cfg, lp, beh, None, torch.rand(len(logp)), w).backward()
    g = lp.grad.abs().mean((0, 2))
    assert g[-2:].max() < 0.01 * g[:-2].min()


def test_turn_penalty_off_is_e5():
    # a budget no turn can exceed keeps every multiplier at 0: the penalty path must then change nothing
    a, _ = train(small_cfg(objective="tree", group_size=2, U=3, qmax=2, drift_budget=10.0), log=False)
    b, _ = train(small_cfg(objective="tree", group_size=2, U=3, qmax=2, drift_budget=10.0, turn_penalty_lr=1e3), log=False)
    for x, y in zip(a.state_dict().values(), b.state_dict().values()):
        assert torch.equal(x, y)
    _, m = train(small_cfg(objective="tree", group_size=2, U=3, qmax=2, drift_budget=1e-6, turn_penalty_lr=1e3), log=False)
    assert max(max(r["lam_by_stal"]) for r in m) > 0  # multipliers move once drift exceeds the budget


def test_scalar_dual_ascent_step():
    _, m = train(small_cfg(objective="coro", U=3, qmax=2, drift_budget=0.05, beta_lr=2.0), log=False)
    for a, b in zip(m, m[1:]):
        assert math.isclose(b["beta"], max(1e-3, a["beta"] + 2.0 * (a["kl_k3"] - 0.05)), rel_tol=1e-9)


def test_beta_path_replays_a_run(tmp_path):
    # control for E7: replaying a run's own beta path must reproduce that run exactly
    kw = dict(objective="tree", group_size=2, U=3, qmax=2, drift_budget=1e-4, beta_lr=5.0, out=str(tmp_path))
    a, ma = train(small_cfg(name="a", **kw))
    b, mb = train(small_cfg(name="b", beta_path="a", **kw))
    assert [r["beta"] for r in ma] == [r["beta"] for r in mb] and len({r["beta"] for r in ma}) > 1
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


def test_tree_flat_sees_the_same_data():
    # control for E5: same forked rollouts as tree, only the loss differs (plain CORO per seed)
    kw = dict(group_size=2, branch=3, iters=1, U=3, qmax=2)
    (a, ma), (b, mb) = (train(small_cfg(objective=o, **kw), log=False) for o in ("tree", "tree_flat"))
    keys = ("env_steps", "branch_turn", "train_return", "kl_k3")
    assert [ma[0][x] for x in keys] == [mb[0][x] for x in keys]
    assert any(not torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


# controls for the Branched CORO mechanism: level 2 alone, and whether siblings win by sharing the user's future
def test_tree_without_level1_is_level2_only():
    from offpolicy import tree_loss
    G1, k, b, H, beta = 4, 3, 2, 5, 0.5
    logp, beh, _ = random_batch(G=G1 * k, n=2, H=H)
    rew = torch.rand(len(logp), H, generator=torch.Generator().manual_seed(1))
    lp = logp.clone().requires_grad_()
    tree_loss(small_cfg(group_size=G1, tree_no_level1=True), lp, beh, rew, (b, k), beta).backward()
    assert (lp.grad[:, :b] == 0).all()
    lp2 = logp[:, b:].clone().requires_grad_()
    (G1 * loss(small_cfg(objective="coro", group_size=k), lp2, beh[:, b:], None, rew[:, b:].sum(1), beta)).backward()
    assert torch.allclose(lp.grad[:, b:], lp2.grad, atol=1e-6)


def test_crn_groups_share_the_shift():
    env = MemoryDialogue(np.repeat(np.arange(8), 4), 16, np.random.default_rng(0), crn=4)
    env.reset()
    for x in (env.t_shift, env.has_shift, env.g_new):
        assert (x.reshape(8, 4) == x.reshape(8, 4)[:, :1]).all()
    assert len(np.unique(env.t_shift)) > 1


def test_fork_resample_redraws_only_the_future():
    B, k, b = 2000, 4, 6
    rng = np.random.default_rng(0)
    env = MemoryDialogue(rng.integers(0, 8, B), 16, rng)  # shift prior: p=0.5, t_shift uniform on 4..11
    env.reset()
    for _ in range(b):
        env.step(turn(B, "PASS", TOK["NONE"]))
    seen = np.repeat(env.has_shift & (env.t_shift <= b), k)
    before = {n: np.repeat(getattr(env, n), k) for n in ("g", "t_shift", "has_shift", "g_new")}
    env.fork(k, resample=True)
    assert all((getattr(env, n)[seen] == x[seen]).all() for n, x in before.items())  # the past is kept
    assert (env.g == before["g"]).all() and not (env.has_shift & (env.t_shift <= b))[~seen].any()
    assert (env.g_new != env.g0).all()
    sib = env.t_shift.reshape(B, k)
    assert (sib != sib[:, :1]).any(1)[~seen[::k]].mean() > 0.9  # siblings no longer share the future
    p = 0.5 * 5 / 8 / (0.5 + 0.5 * 5 / 8)  # P(shift still to come | none by turn 6)
    assert abs(env.has_shift[~seen].mean() - p) < 0.03


def test_controls_train():
    train(small_cfg(objective="coro", crn=True), log=False)
    train(small_cfg(objective="tree", group_size=2, fork_resample=True, tree_no_level1=True), log=False)


# 6. env bounds: scripted policies that only read their context
def run_scripted(policy_fn, H, W, B=4000, seed=0, **env_kw):
    rng = np.random.default_rng(seed)
    env = MemoryDialogue(rng.integers(0, 8, B), H, rng, **env_kw)
    obs, obs_hist, act_hist, total = env.reset(), [], [], np.zeros(B)
    for t in range(H):
        obs_hist.append(obs)
        a = policy_fn(t, H, build_context(env.mem, obs_hist, act_hist, W), rng)
        obs, r, _ = env.step(a)
        total += r
        act_hist.append(np.concatenate([np.full((B, 1), TOK["<agent>"]), a], 1))
    return total.mean()


def turn(B, verb, arg):
    return np.stack(np.broadcast_arrays(np.full(B, TOK[verb]), arg), 1)


def latest_goal_in_context(ctx):
    """Goal token after the last RESULT or NOW in the visible context, else G0 (a blind guess)."""
    hit = np.isin(ctx[:, :-1], [TOK["RESULT"], TOK["NOW"]])
    last = ctx.shape[1] - 2 - hit[:, ::-1].argmax(1)
    return np.where(hit.any(1), ctx[np.arange(len(ctx)), last + 1], GOAL_IDS[0])


def memory_oracle(t, H, ctx, rng, follow_shift=True):
    obs_word, obs_arg = ctx[:, -3], ctx[:, -2]
    if t == 0:
        return turn(len(ctx), "LOOKUP", TOK["NONE"])
    if t == H - 1:
        return turn(len(ctx), "ANSWER", ctx[:, 1])  # memory slot
    news = (obs_word == TOK["RESULT"]) | (follow_shift & (obs_word == TOK["NOW"]))
    return np.where(news[:, None], turn(len(ctx), "WRITE", obs_arg), turn(len(ctx), "PASS", TOK["NONE"]))


def context_only(t, H, ctx, rng):
    if t == 0:
        return turn(len(ctx), "LOOKUP", TOK["NONE"])
    if t == H - 1:
        return turn(len(ctx), "ANSWER", latest_goal_in_context(ctx))
    return turn(len(ctx), "PASS", TOK["NONE"])


def answers_early(t, H, ctx, rng):
    if t == 0:
        return turn(len(ctx), "LOOKUP", TOK["NONE"])
    if t == H - 1:
        return turn(len(ctx), "PASS", TOK["NONE"])  # final reply carries no goal
    return turn(len(ctx), "ANSWER", latest_goal_in_context(ctx))


def uniform(t, H, ctx, rng):
    return np.stack([rng.choice(VERB_IDS, len(ctx)), rng.choice(ARG_IDS, len(ctx))], 1)


def test_env_bounds():
    assert run_scripted(memory_oracle, H=64, W=8) == 1.0  # tools and PASS are free
    ignore = lambda *a: memory_oracle(*a, follow_shift=False)  # noqa: E731
    assert 0.45 < run_scripted(ignore, H=64, W=8) < 0.55  # misses the shift half the time
    assert run_scripted(context_only, H=16, W=16) == 1.0
    assert 0.09 < run_scripted(context_only, H=64, W=8) < 0.16  # goal scrolled out: 1/8 guess
    assert abs(run_scripted(answers_early, H=16, W=16) + 0.14) < 1e-6  # 14 unasked answers, no reward
    assert 0.03 < run_scripted(uniform, H=16, W=16) < 0.07  # 1/9 goal minus ~0.06 cost
