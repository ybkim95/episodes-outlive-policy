"""Main-matrix summary from runs/m_*/metrics.jsonl only.

usage: python analyze.py   -> prints a markdown table, writes figs/main_curves.png and figs/main_drift.png
final = mean held-out goal rate over the last 3 evals; auc = mean over all evals (sample efficiency).
Every seed is listed; nothing is dropped.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).parent
LABEL = {
    "E0_coro_sync": "E0 CORO, no staleness",
    "E1_ppo": "E1 PPO-clip (token IS)",
    "E1b_naive": "E1b ignore staleness",
    "E2_oapl": "E2 OAPL",
    "E3_coro": "E3 CORO",
    "E4_coro_budget": "E4 CORO + drift budget",
    "E5_tree_budget": "E5 Branched CORO + budget",
    "E6_tree_turnbudget": "E6 Branched CORO + per-turn budget",
    "E7_tree_turnpenalty": "E7 Branched CORO + per-turn KL penalty",
}
E7_LRS = (100, 1000, 10000)  # multiplier step sizes in scripts/e7.sh


def load():
    runs = defaultdict(dict)
    for d in sorted((HERE / "runs").glob("m_*_s*")):
        cfg, seed = re.match(r"m_(.*)_s(\d+)$", d.name).groups()
        rows = [json.loads(line) for line in (d / "metrics.jsonl").read_text().splitlines()]
        runs[cfg][int(seed)] = rows
    lr = select_e7()
    if lr is not None:
        for seed in (0, 1, 2):
            runs["E7_tree_turnpenalty"][seed] = read(f"e7_lr{lr}_s{seed}")
    return runs


def read(name):
    return [json.loads(line) for line in (HERE / "runs" / name / "metrics.jsonl").read_text().splitlines()]


def select_e7():
    """Multiplier step size for E7, chosen on tuning seed 100 only: best final goal, then best AUC."""
    score = {}
    for lr in E7_LRS:
        g = final_goal(f"e7_lr{lr}", 100)
        if g is None:
            return None  # sweep not finished
        score[lr] = (round(g, 2), evals(read(f"e7_lr{lr}_s100"))[1].mean())
    return max(score, key=score.get)


def e7():
    """Every step size on every seed, so the seed-100 choice can be checked; plus where the drift went."""
    sel = select_e7()
    if sel is None:
        print("\nE7 sweep not finished")
        return
    print(f"\n| E7 multiplier step (chosen: {sel}) | seed 100 (tuning) | seeds 0, 1, 2 | mean 0-2 |")
    print("|---|---|---|---|")
    for lr in E7_LRS:
        v = [final_goal(f"e7_lr{lr}", s) for s in (100, 0, 1, 2)]
        print(f"| {lr} | {v[0]:.2f} | " + ", ".join(f"{x:.2f}" for x in v[1:]) + f" | {np.mean(v[1:]):.2f} |")
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.2))
    curves = [("E5 (no per-turn control)", "m_E5_tree_budget"), ("E6 per-turn beta", "m_E6_tree_turnbudget")]
    curves += [(f"E7 step {lr}", f"e7_lr{lr}") for lr in E7_LRS]
    for i, (lab, name) in enumerate(curves):
        x, _ = evals(read(f"{name}_s0"))
        axs[0].plot(x, np.mean([evals(read(f"{name}_s{s}"))[1] for s in (0, 1, 2)], 0), color=f"C{i}", label=lab)
    axs[0].set(title="held-out goal rate (mean of seeds 0-2)", xlabel="agent turns generated", ylim=(0, 1.02))
    axs[0].legend(fontsize=6)
    kl = np.array([[np.nan if x is None else x for x in r["kl_by_stal"]] for r in read(f"e7_lr{sel}_s0")])
    for b in range(kl.shape[1]):
        axs[1].plot(np.convolve(np.nan_to_num(kl[:, b]), np.ones(20) / 20, "valid"), lw=0.8,
                    color=plt.cm.viridis(b / max(1, kl.shape[1] - 1)), label=f"stale {b}")
    axs[1].axhline(0.002, color="k", ls=":", lw=1)
    axs[1].set(title=f"E7 (step {sel}, seed 0): per-turn KL by staleness", xlabel="iter", yscale="log", ylim=(1e-7, 1e-2))
    axs[1].legend(fontsize=5, ncol=2)
    lam = np.array([r["lam_by_stal"] for r in read(f"e7_lr{sel}_s0")])
    for b in range(lam.shape[1]):
        axs[2].plot(lam[:, b], lw=0.8, color=plt.cm.viridis(b / max(1, lam.shape[1] - 1)))
    axs[2].set(title="E7: KL multiplier per staleness bucket", xlabel="iter")
    fig.tight_layout()
    fig.savefig(HERE / "figs/e7_drift.png", dpi=150)


def evals(rows, key="eval_goal"):
    ev = [r for r in rows if key in r]
    return np.array([r["env_steps"] for r in ev]), np.array([r[key] for r in ev])


def table(runs):
    print("| config | iters | final goal (per seed) | mean final | mean AUC | M turns to 0.9 (seeds reaching) |")
    print("|---|---|---|---|---|---|")
    for cfg in LABEL:
        if cfg not in runs:
            continue
        finals, aucs, its, t90 = [], [], [], []
        for s, rows in sorted(runs[cfg].items()):
            x, y = evals(rows)
            if (y >= 0.9).any():
                t90.append(x[np.argmax(y >= 0.9)] / 1e6)
            finals.append(y[-3:].mean())
            aucs.append(y.mean())
            its.append(rows[-1]["iter"] + 1)
        seeds = ", ".join(f"{f:.2f}" for f in finals)
        print(f"| {LABEL[cfg]} | {min(its)} | {seeds} | {np.mean(finals):.2f} | {np.mean(aucs):.2f} | {f"{np.mean(t90):.2f} ({len(t90)}/{len(finals)})" if t90 else "-"} |")


def curves(runs):
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, cfg in enumerate(c for c in LABEL if c in runs):
        col = f"C{i}"
        ys = []
        for rows in runs[cfg].values():
            x, y = evals(rows)
            ax.plot(x, y, color=col, alpha=0.2, lw=1)
            ys.append((x, y))
        n = min(len(y) for _, y in ys)
        ax.plot(ys[0][0][:n], np.mean([y[:n] for _, y in ys], 0), color=col, lw=2, label=f"{LABEL[cfg]} (n={len(ys)})")
    ax.axhline(1 / 8, color="gray", ls=":", lw=1)
    ax.set(xlabel="agent turns generated", ylabel="held-out goal rate", ylim=(0, 1.02), title="H=16, staleness 0..11 within episodes")
    ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(HERE / "figs/main_curves.png", dpi=150)


def drift(runs):
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.2))
    for i, cfg in enumerate(c for c in LABEL if c in runs):
        rows = runs[cfg][min(runs[cfg])]  # first seed, for readability
        it = [r["iter"] for r in rows]
        axs[0].plot(it, [r["ess_traj"] for r in rows], color=f"C{i}", lw=0.8, label=LABEL[cfg])
        axs[1].plot(it, [r["kl_k3"] for r in rows], color=f"C{i}", lw=0.8)
    axs[0].set(title="trajectory-weight ESS / B", xlabel="iter", yscale="log")
    axs[1].set(title="per-token KL(behaviour || trainer)", xlabel="iter", yscale="log")
    axs[0].legend(fontsize=5.5)
    if "E6_tree_turnbudget" in runs:
        rows = runs["E6_tree_turnbudget"][min(runs["E6_tree_turnbudget"])]
        b = np.array([r["beta_by_stal"] for r in rows])
        for s in range(b.shape[1]):
            axs[2].plot(b[:, s], color=plt.cm.viridis(s / max(1, b.shape[1] - 1)), lw=0.8, label=f"stale {s}")
        axs[2].set(title="E6: beta per staleness bucket", xlabel="iter")
        axs[2].legend(fontsize=5, ncol=2)
    fig.tight_layout()
    fig.savefig(HERE / "figs/main_drift.png", dpi=150)


SEVERITY = [  # (label, run-name pattern per objective); the mildest level is the main matrix
    ("U=4, q<=8\n(stale <= 11)", {"ppo": "m_E1_ppo", "oapl": "m_E2_oapl", "coro": "m_E3_coro", "none": "m_E1b_naive"}),
    ("U=1, q<=16\n(stale <= 31)", {o: f"sev_q16_{o}" for o in ("ppo", "oapl", "coro", "none")}),
    ("U=1, q<=48\n(stale <= 63)", {o: f"sev_q48_{o}" for o in ("ppo", "oapl", "coro", "none")}),
]
SEV_LABEL = {"ppo": "PPO-clip (token IS)", "oapl": "OAPL", "coro": "CORO", "none": "ignore staleness"}


def final_goal(name, seed):
    p = HERE / "runs" / f"{name}_s{seed}" / "metrics.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    if rows[-1]["iter"] + 1 < 1000:
        return None  # unfinished
    return evals(rows)[1][-3:].mean()


def severity():
    print("\n| objective | " + " | ".join(lab.replace("\n", " ") for lab, _ in SEVERITY) + " |")
    print("|---|" + "---|" * len(SEVERITY))
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for i, obj in enumerate(SEV_LABEL):
        cells, means = [], []
        for _, names in SEVERITY:
            v = [final_goal(names[obj], s) for s in (0, 1, 2)]
            v = [x for x in v if x is not None]
            cells.append(", ".join(f"{x:.2f}" for x in v) or "-")
            means.append(np.mean(v) if v else np.nan)
            for x in v:
                ax.scatter(len(means) - 1 + 0.04 * (i - 1.5), x, color=f"C{i}", s=10, alpha=0.5)
        ax.plot(range(len(means)), means, color=f"C{i}", lw=2, marker="o", label=SEV_LABEL[obj])
        print(f"| {SEV_LABEL[obj]} | " + " | ".join(cells) + " |")
    ax.set_xticks(range(len(SEVERITY)), [lab for lab, _ in SEVERITY], fontsize=7)
    ax.axhline(1 / 8, color="gray", ls=":", lw=1)
    ax.set(ylabel="final held-out goal rate (3 seeds)", ylim=(0, 1.05), title="H=16: how stale can the data get?")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(HERE / "figs/severity.png", dpi=150)


if __name__ == "__main__":
    (HERE / "figs").mkdir(exist_ok=True)
    r = load()
    table(r)
    curves(r)
    drift(r)
    severity()
    e7()
