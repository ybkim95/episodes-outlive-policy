"""Real asynchronous sampler and trainer: separate processes, a bounded queue, weight sync at turn boundaries.

runner.py fakes asynchrony with a ring buffer of past weights so staleness can be set exactly. Here nobody
sets it. N sampler processes (CPU) each generate whole batches with whatever weights they last pulled; the
trainer (GPU) trains on whichever batch arrives next and publishes new weights after every update. A
sampler swaps weights only at turn boundaries and writes the version it used into every turn, so per-turn
staleness is measured: staleness = trainer version at the update - version that sampled the turn.
The record (traj.Rollout) and the losses (offpolicy.py) are the same code runner.py uses.

turn_delay stands in for the user model and tools, the expensive part of a real turn. max_age bounds the
age of a batch's LAST turn (older turns in it are kept: their versions are recorded); staler batches are
dropped and counted. Behaviour log-probs come from the sampler's CPU forward and are scored by the trainer
on GPU, so even a turn with staleness 0 has a small numeric log-ratio, as with a real inference engine.
Only global-beta objectives (ppo, coro, tree, optionally with the drift budget) are supported.
"""
import hashlib
import json
import queue as pyqueue
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.multiprocessing as mp
from torch import nn

from credit import CREDIT
from env import N_GOALS
from offpolicy import drift_stats, loss, tree_loss
from policy import Policy
from runner import HERE, Config, backward_two_pass, evaluate, parse_args, render, rollout, src_hash
from traj import max_context_len


@dataclass
class AsyncConfig(Config):
    samplers: int = 8
    queue: int = 2  # batches the queue holds; a sampler blocks when it is full
    turn_delay: float = 0.0  # seconds per turn spent outside the policy (user model, tools)
    max_age: int = 32  # drop a batch whose last turn is more than this many updates stale


def copy_into(model, flat):
    with torch.no_grad():
        i = 0
        for p in model.parameters():
            n = p.numel()
            p.copy_(flat[i : i + n].view_as(p))
            i += n


def publish(model, flat, version, lock, v):
    vec = nn.utils.parameters_to_vector(model.parameters()).detach().cpu()
    with lock:
        flat.copy_(vec)
        version.value = v


def sample_batch(cfg, model, pull, rng, gen):
    """One batch of episodes. pull() refreshes model's weights if newer ones exist and returns their
    version; it runs at every turn boundary, after turn_delay. Returns (rollout with absolute per-turn
    versions, branch)."""
    B, H = cfg.groups * cfg.group_size, cfg.H
    goals = np.repeat(rng.integers(0, N_GOALS, cfg.groups), cfg.group_size)
    branch = (int(rng.integers(1, H)), cfg.branch) if cfg.objective == "tree" else (0, 1)
    ver = np.zeros(H, np.int64)

    def before_turn(t):
        if t and cfg.turn_delay:
            time.sleep(cfg.turn_delay)  # the user's or tool's reply to turn t-1
        ver[t] = pull()

    ro = rollout(lambda s: model, cfg, goals, rng, gen, np.zeros((B, H), int), branch, before_turn=before_turn)
    ro.version = np.repeat(ver[None], len(ro.act), 0)  # lockstep batch: every row swaps at the same turn
    return ro, branch


def sampler(cfg, rank, flat, version, lock, q, stop):
    torch.set_num_threads(1)
    rng = np.random.default_rng([cfg.seed, 7, rank])
    gen = torch.Generator().manual_seed(cfg.seed * 1000 + rank)
    model = Policy(max_context_len(cfg.H, cfg.W))
    have = [-1]

    def pull():
        if version.value != have[0]:
            with lock:
                copy_into(model, flat)
                have[0] = version.value
        return have[0]

    parent = mp.parent_process()
    running = lambda: not stop.is_set() and parent.is_alive()  # noqa: E731  (exit if the trainer was killed)
    while running():
        ro, branch = sample_batch(cfg, model, pull, rng, gen)
        item = (ro, branch, time.time(), rank)
        while running():
            try:
                q.put(item, timeout=1.0)
                break
            except pyqueue.Full:
                pass


def file_hash(name):
    return hashlib.sha256((HERE / name).read_bytes()).hexdigest()[:12]


def train_async(cfg, log=True):
    assert cfg.objective in ("ppo", "coro", "tree") and not (cfg.drift_per_turn or cfg.turn_penalty_lr or cfg.beta_path)
    assert not (cfg.sync or cfg.lag or cfg.U or cfg.qmax or cfg.crn or cfg.fork_resample or cfg.tree_no_level1)
    torch.manual_seed(cfg.seed)  # same initial weights as runner.py for the same seed
    dev = torch.device(cfg.device)
    model = Policy(max_context_len(cfg.H, cfg.W)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    H, beta, tree = cfg.H, cfg.beta, cfg.objective == "tree"

    run_dir = None
    if log:
        run_dir = HERE / cfg.out / (cfg.name or f"async_{cfg.objective}_n{cfg.samplers}_s{cfg.seed}")
        run_dir.mkdir(parents=True, exist_ok=True)
        meta = {**asdict(cfg), "src_hash": src_hash(), "async_hash": file_hash("async_runner.py"), "torch": torch.__version__}
        (run_dir / "config.json").write_text(json.dumps(meta, indent=1))
        (run_dir / "metrics.jsonl").write_text("")

    ctx = mp.get_context("spawn")
    flat = torch.zeros(sum(p.numel() for p in model.parameters())).share_memory_()
    version, lock, q, stop = ctx.Value("l", -1), ctx.Lock(), ctx.Queue(cfg.queue), ctx.Event()
    publish(model, flat, version, lock, 0)
    procs = [ctx.Process(target=sampler, args=(cfg, r, flat, version, lock, q, stop), daemon=True) for r in range(cfg.samplers)]
    for p in procs:
        p.start()

    metrics, t0, env_steps, dropped, it = [], time.time(), 0, 0, 0
    try:
        while it < cfg.iters:
            tw = time.time()
            ro, branch, t_put, rank = q.get()
            wait, queue_age = time.time() - tw, time.time() - t_put
            stal = it - ro.version  # (B', H): measured, not set
            assert stal.min() >= 0
            if stal[:, -1].max() > cfg.max_age:
                dropped += 1
                continue
            tb = time.time()
            beh = torch.as_tensor(ro.logp, device=dev)
            if tree:
                rew = torch.as_tensor(ro.reward, device=dev)
                loss_of = lambda lp: tree_loss(cfg, lp, beh, rew, branch, beta)  # noqa: E731
            else:
                adv = torch.as_tensor(CREDIT[cfg.credit](ro, cfg), device=dev)
                ret = torch.as_tensor(ro.ret, device=dev)
                loss_of = lambda lp: loss(cfg, lp, beh, adv, ret, beta)  # noqa: E731
            lossv, logp = backward_two_pass(model, ro, loss_of)
            gn = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            publish(model, flat, version, lock, it + 1)

            B = cfg.groups * cfg.group_size
            env_steps += B * branch[0] + B * branch[1] * (H - branch[0])
            lr_tok = (logp - beh).detach()
            kl_turn = (lr_tok.exp() - 1 - lr_tok).mean(2).cpu().numpy()
            kl_by_stal = [float(kl_turn[stal == s].mean()) if (stal == s).any() else None for s in range(int(stal.max()) + 1)]
            m = dict(iter=it, env_steps=env_steps, staleness=float(stal.mean()), staleness_max=int(stal.max()),
                     stal_first=float(stal[:, 0].mean()), stal_last=float(stal[:, -1].mean()),
                     versions_per_episode=int(len(np.unique(ro.version[0]))), branch_turn=branch[0], sampler=rank,
                     wait=wait, queue_age=queue_age, busy=time.time() - tb, dropped=dropped, wall=time.time() - t0,
                     loss=lossv.item(), grad_norm=gn.item(), beta=float(beta), kl_by_stal=kl_by_stal,
                     train_return=float(ro.ret.mean()), train_goal=float(ro.reward[:, -1].mean()),
                     **drift_stats(logp, beh, cfg.group_size))
            if cfg.drift_budget > 0:
                beta = max(1e-3, beta + cfg.beta_lr * (m["kl_k3"] - cfg.drift_budget))
            if it % cfg.eval_every == 0 or it == cfg.iters - 1:
                ev = evaluate(model, cfg)
                m["eval_return"], m["eval_goal"] = float(ev.ret.mean()), float(ev.reward[:, -1].mean())
                if log:
                    print(f"it={it:4d} steps={env_steps:>9d} goal={m['eval_goal']:.3f} stal={m['staleness']:.1f} "
                          f"(first {m['stal_first']:.1f}, last {m['stal_last']:.1f}, max {m['staleness_max']}) "
                          f"wait={wait:.2f}s busy={m['busy']:.2f}s kl={m['kl_k3']:.1e} beta={beta:.3f} {m['wall']:.0f}s", flush=True)
            metrics.append(m)
            if log:
                with open(run_dir / "metrics.jsonl", "a") as f:
                    f.write(json.dumps(m) + "\n")
            it += 1
    finally:
        stop.set()
        while True:  # unblock samplers stuck in put()
            try:
                q.get_nowait()
            except (pyqueue.Empty, OSError, ValueError):
                break
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()

    if log:
        ev = evaluate(model, cfg, n=8)
        (run_dir / "trajectories.txt").write_text("\n\n".join(render(ev, i) for i in range(8)))
    return model, metrics


if __name__ == "__main__":
    train_async(parse_args(AsyncConfig))
