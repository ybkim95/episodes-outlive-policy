"""Main-matrix summary from runs/m_*/metrics.jsonl only.

usage: python analyze.py   -> prints a markdown table, writes figs/main_curves.png and figs/main_drift.png
final = mean held-out goal rate over the last 3 evals; auc = mean over all evals (sample efficiency).
Every seed is listed; nothing is dropped.
"""
import itertools
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


def select_e7(pre="e7"):
    """Multiplier step size for E7 (or X1), chosen on tuning seed 100 only: best final goal, then best AUC."""
    score = {}
    for lr in E7_LRS:
        g = final_goal(f"{pre}_lr{lr}", 100)
        if g is None:
            return None  # sweep not finished
        score[lr] = (round(g, 2), evals(read(f"{pre}_lr{lr}_s100"))[1].mean())
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


def drift_table():
    """Full-run drift, every iteration of every seed (no window). ESS is per update, over the B episodes;
    'beta floor' is the first iteration a scalar drift budget pinned beta at 1e-3, per seed."""
    print("\n| config | median ESS | 5th pct ESS | min ESS | median max ratio | max ratio | beta floor (iter, per seed) |")
    print("|---|---|---|---|---|---|---|")
    groups = [(LABEL[c], [f"m_{c}_s{s}" for s in (0, 1, 2)]) for c in LABEL if c != "E7_tree_turnpenalty"]
    groups += [("E7 (step 100)", [f"e7_lr100_s{s}" for s in (0, 1, 2)])]
    groups += [(f"{SEV_LABEL[o]}, stale <= {n}", [f"sev_q{q}_{o}_s{s}" for s in (0, 1, 2)])
               for q, n in ((16, 31), (48, 63)) for o in ("ppo", "oapl", "coro")]
    for label, names in groups:
        names = [n for n in names if (HERE / "runs" / n / "metrics.jsonl").exists()]
        if not names:
            continue
        rows = [read(n) for n in names]
        ess = np.concatenate([[r["ess_traj"] for r in rr] for rr in rows])
        rmax = np.concatenate([[r["ratio_max"] for r in rr] for rr in rows])
        floor = [next((r["iter"] for r in rr if r["beta"] <= 1.0001e-3), None) for rr in rows]
        fl = ", ".join("-" if f is None else str(f) for f in floor) if any(f is not None for f in floor) else ""
        print(f"| {label} | {np.median(ess):.2f} | {np.percentile(ess, 5):.2f} | {ess.min():.3f} | "
              f"{np.median(rmax):.2f} | {rmax.max():.1f} | {fl} |")


CONTROLS = [  # (label, run prefix); every run is seeds 0-2 at 1000 updates; see scripts/controls.sh
    ("E4 CORO + budget, 256 episodes/update", "r_E4_coro_budget"),
    ("C2 CORO + budget, 416 episodes/update", "c_coro416"),
    ("C1 forked data, flat CORO (416 leaves)", "c_tree_flat"),
    ("E5 forked data, Branched CORO (416 leaves)", "r_E5_tree_budget"),
    ("E7 Branched CORO + per-bucket penalty", "e7_lr100"),
    ("C3 E7 with E5's beta path", "c_e7_betaE5"),
    ("C4 E5 with E7's beta path", "c_e5_betaE7"),
    ("E0 sync CORO", "m_E0_coro_sync"),
    ("C5 sync policy gradient (= sync PPO)", "c_sync_none"),
    ("M1 E5, level 2 only", "k_tree_lvl2"),  # scripts/mechanism.sh
    ("M2 E5, siblings redraw the pending goal change", "k_tree_resample"),
    ("M3 E4, each group shares one goal-change schedule", "k_coro_crn"),
    ("M1p M1 paired, E5's beta path replayed", "v_tree_lvl2_bp"),  # scripts/review.sh
    ("M2p M2 paired, E5's beta path replayed", "v_tree_resample_bp"),
    ("M3p M3 paired, E4's beta path replayed", "v_coro_crn_bp"),
    ("E1 PPO-clip, flat, no budget", "m_E1_ppo"),
    ("Branched CORO, no budget (beta 0.3 fixed)", "f_tree_beta0.3"),
    ("B1 forked data, branched PPO-clip, no budget", "v_tree_ppo"),
    ("S1 sync CORO, 4 minibatch steps per batch", "v_sync_coro_mb4"),
    ("S2 sync PPO-clip, 4 minibatch steps per batch", "v_sync_ppo_mb4"),
    ("S3 sync policy gradient, 4 minibatch steps per batch", "v_sync_none_mb4"),
]


def first_at(rows, key, thr=0.9):
    ev = [r for r in rows if "eval_goal" in r]
    hit = next((r for r in ev if r["eval_goal"] >= thr), None)
    return None if hit is None else hit[key] + (1 if key == "iter" else 0)


def controls():
    """Is E5's speed the loss, the forked data, or just more episodes per update? Is E7 slow because of its
    penalty or because its beta decayed early? Updates to 0.9 has the eval resolution (every 50 updates)."""
    print("\n| config | final goal (per seed) | updates to 0.9 (per seed) | M turns to 0.9 | k episode endings to 0.9 |")
    print("|---|---|---|---|---|")
    out = {}
    for label, pre in CONTROLS:
        names = [f"{pre}_s{s}" for s in (0, 1, 2) if (HERE / "runs" / f"{pre}_s{s}" / "metrics.jsonl").exists()]
        if not names:
            continue
        rows = [read(n) for n in names]
        cfg = json.loads((HERE / "runs" / names[0] / "config.json").read_text())
        leaves = cfg["groups"] * cfg["group_size"] * (cfg["branch"] if cfg["objective"].startswith("tree") else 1)
        fin = [evals(rr)[1][-3:].mean() for rr in rows]
        its = [first_at(rr, "iter") for rr in rows]
        turns = [first_at(rr, "env_steps") for rr in rows]
        f = lambda xs, k=1: ", ".join("-" if x is None else f"{x / k:.0f}" if k == 1 else f"{x / k:.2f}" for x in xs)  # noqa: E731
        done = rows[0][-1]["iter"] + 1
        print(f"| {label}{'' if done == 1000 else f' (at {done})'} | {', '.join(f'{x:.2f}' for x in fin)} | {f(its)} | {f(turns, 1e6)} | "
              f"{f([None if i is None else i * leaves for i in its], 1e3)} |")
        out[pre] = rows
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, pres in ((axs[0], ("r_E4_coro_budget", "c_coro416", "c_tree_flat", "r_E5_tree_budget")),
                     (axs[1], ("r_E5_tree_budget", "e7_lr100", "c_e7_betaE5", "c_e5_betaE7"))):
        for i, pre in enumerate(p for p in pres if p in out):
            ys = [evals(rr) for rr in out[pre]]
            n = min(len(y) for _, y in ys)
            it = [r["iter"] for r in out[pre][0] if "eval_goal" in r][:n]
            ax.plot(it, np.mean([y[:n] for _, y in ys], 0), color=f"C{i}", lw=2, label=dict((p, lab) for lab, p in CONTROLS)[pre])
            for _, y in ys:
                ax.plot(it, y[:n], color=f"C{i}", lw=0.6, alpha=0.3)
        ax.axhline(0.9, color="gray", ls=":", lw=1)
        ax.set(xlabel="updates", ylabel="held-out goal rate", ylim=(0, 1.02))
        ax.legend(fontsize=6.5, loc="lower right")
    axs[0].set_title("What makes Branched CORO faster per update?")
    axs[1].set_title("Is E7 slow because of its beta path?")
    fig.tight_layout()
    fig.savefig(HERE / "figs/controls.png", dpi=150)
    for a, b in (("m_E4_coro_budget", "r_E4_coro_budget"), ("m_E5_tree_budget", "r_E5_tree_budget")):
        for s in (0, 1, 2):
            if (HERE / "runs" / f"{b}_s{s}").exists():
                x, y = read(f"{a}_s{s}"), read(f"{b}_s{s}")
                same = next((i for i, (u, v) in enumerate(zip(x, y)) if u["loss"] != v["loss"]), min(len(x), len(y)))
                print(f"provenance {a} vs rerun, seed {s}: identical loss for the first {same} of {min(len(x), len(y))} updates")


BETA_SWEEP = {  # fixed beta from the first update, no drift budget; see scripts/followup.sh
    "flat CORO": {0.001: "f_coro_beta0.001", 0.03: "f_coro_beta0.03", 0.3: "m_E3_coro", 3: "f_coro_beta3"},
    "Branched CORO": {0.001: "f_tree_beta0.001", 0.03: "f_tree_beta0.03", 0.3: "f_tree_beta0.3", 3: "f_tree_beta3"},
}
H32 = [("E4 CORO + budget", "h32_E4_coro_budget"), ("C1 forked data, flat CORO", "h32_C1_tree_flat"),
       ("E5 Branched CORO + budget", "h32_E5_tree_budget")]


def followup():
    """Does beta matter when it is fixed from the start? Does the ranking hold at 32 turns?"""
    print("\n| objective | beta | final goal (per seed) | updates to 0.9 (per seed) | median beta x within-group sd of l |")
    print("|---|---|---|---|---|")
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    for i, (obj, runs) in enumerate(BETA_SWEEP.items()):
        xs, fins, its = [], [], []
        for b, pre in runs.items():
            rows = [read(f"{pre}_s{s}") for s in (0, 1, 2) if (HERE / "runs" / f"{pre}_s{s}" / "metrics.jsonl").exists()]
            if not rows:
                continue
            fin = [evals(rr)[1][-3:].mean() for rr in rows]
            it = [first_at(rr, "iter") for rr in rows]
            spread = np.median([b * r["ell_std"] for rr in rows for r in rr])
            done = min(rr[-1]["iter"] + 1 for rr in rows)
            print(f"| {obj} | {b} | {', '.join(f'{x:.2f}' for x in fin)}{'' if done == 1000 else f' (at {done})'} | "
                  f"{', '.join('-' if x is None else str(x) for x in it)} | {spread:.3f} |")
            for f, t in zip(fin, it):
                xs.append(b)
                fins.append(f)
                its.append(1000 if t is None else t)  # never reached: plotted at the run length, so the mean is a lower bound
        for ax, ys in ((axs[0], fins), (axs[1], its)):
            ax.scatter(np.array(xs) * (1.08 if i else 0.93), ys, color=f"C{i}", s=14, alpha=0.6)
            ax.plot(sorted(set(xs)), [np.mean([y for x, y in zip(xs, ys) if x == b]) for b in sorted(set(xs))],
                    color=f"C{i}", lw=2, marker="o", label=obj)
    axs[0].set(xscale="log", xlabel="fixed beta", ylabel="final held-out goal rate", ylim=(0, 1.05))
    axs[1].axhline(1000, color="gray", ls=":", lw=1)
    axs[1].text(4e-3, 990, "never reached (run length)", fontsize=7, color="gray", va="top")
    axs[1].set(xscale="log", xlabel="fixed beta", ylabel="updates to 0.9", ylim=(250, 1030))
    axs[0].legend(fontsize=7)
    fig.suptitle("Staleness <= 11, no drift budget, 3 seeds per point", fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / "figs/beta_sweep.png", dpi=150)
    print("\n| H = 32 | final goal (per seed) | updates to 0.5 (per seed) | updates to 0.9 (per seed) | M turns to 0.9 |")
    print("|---|---|---|---|---|")
    for label, pre in H32:
        rows = [read(f"{pre}_s{s}") for s in (0, 1, 2) if (HERE / "runs" / f"{pre}_s{s}" / "metrics.jsonl").exists()]
        if rows:
            done = min(rr[-1]["iter"] + 1 for rr in rows)
            turns = [first_at(rr, "env_steps") for rr in rows]
            print(f"| {label}{'' if done == 1000 else f' (at {done})'} | {', '.join(f'{evals(rr)[1][-3:].mean():.2f}' for rr in rows)} | "
                  f"{', '.join('-' if x is None else str(x) for x in (first_at(rr, 'iter', 0.5) for rr in rows))} | "
                  f"{', '.join('-' if x is None else str(x) for x in (first_at(rr, 'iter') for rr in rows))} | "
                  f"{', '.join('-' if x is None else f'{x / 1e6:.2f}' for x in turns)} |")


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


def seeds(pre):
    return [read(f"{pre}_s{s}") for s in (0, 1, 2) if (HERE / "runs" / f"{pre}_s{s}" / "metrics.jsonl").exists()]


def fixes():
    """X1: the fix the talk proposed for per-turn budgets: E7 with beta held at 0.3, so only the per-bucket KL
    multipliers move. Step size chosen on seed 100 as for E7. X2: is branching's gain specific to stale data?"""
    sel = select_e7("x_e7fix")
    print(f"\n| X1 step | seed 100 (tuning) | seeds 0, 1, 2 | updates to 0.9 (seeds 0-2) |")
    print("|---|---|---|---|")
    for lr in E7_LRS if sel is not None else ():
        v = [final_goal(f"x_e7fix_lr{lr}", s) for s in (100, 0, 1, 2)]
        its = [first_at(read(f"x_e7fix_lr{lr}_s{s}"), "iter") for s in (0, 1, 2)]
        print(f"| {lr}{' (chosen)' if lr == sel else ''} | {v[0]:.2f} | " + ", ".join(f"{x:.2f}" for x in v[1:])
              + " | " + ", ".join("-" if x is None else str(x) for x in its) + " |")
    print("\n| config | final goal (per seed) | updates to 0.9 (per seed) |")
    print("|---|---|---|")
    rows = [("E0 CORO, no staleness", "m_E0_coro_sync"), ("X2 Branched CORO, no staleness", "x_tree_sync"),
            ("E3 CORO, stale, no budget", "m_E3_coro"), ("Branched CORO, stale, no budget (beta 0.3)", "f_tree_beta0.3")]
    if sel is not None:
        rows += [("E7 per-turn penalty, beta by dual ascent", "e7_lr100"), (f"X1 per-turn penalty, beta fixed (step {sel})", f"x_e7fix_lr{sel}"),
                 (f"X1g one global penalty, beta fixed (step {sel}, not retuned)", f"x_e7glob_lr{sel}")]
    for label, pre in rows:
        rr = seeds(pre)
        if rr:
            done = min(r[-1]["iter"] + 1 for r in rr)
            print(f"| {label}{'' if done == 1000 else f' (at {done})'} | {', '.join(f'{evals(r)[1][-3:].mean():.2f}' for r in rr)} | "
                  + ", ".join("-" if x is None else str(x) for x in (first_at(r, "iter") for r in rr)) + " |")


def done_seeds(pre, n=8):
    """Finished runs only (1000 updates), by seed."""
    out = {}
    for s in range(n):
        p = HERE / "runs" / f"{pre}_s{s}" / "metrics.jsonl"
        if p.exists():
            rows = read(f"{pre}_s{s}")
            if rows[-1]["iter"] + 1 == 1000:
                out[s] = rows
    return out


def rank_test(a, b):
    """Exact two-sided permutation test on the mean rank (Mann-Whitney), over every split of the pooled runs."""
    x = np.r_[a, b].astype(float)
    rk = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)
    for v in np.unique(x):  # average ranks for ties
        rk[x == v] = rk[x == v].mean()
    obs = abs(rk[: len(a)].mean() - rk[len(a):].mean())
    d = [abs(rk[list(c)].mean() - np.delete(rk, c).mean()) for c in itertools.combinations(range(len(x)), len(a))]
    return float(np.mean(np.array(d) >= obs - 1e-9))


EIGHT = [  # (label, run prefix, episodes per update); scripts/review2.sh, eval every 10 updates
    ("E4 CORO + budget", "n_E4", 256), ("T256 Branched CORO + budget", "n_T256", 256),
    ("E5 Branched CORO + budget", "n_E5", 416), ("C1 flat CORO on E5's forked data", "n_C1", 416),
    ("C2 CORO + budget, independent episodes", "n_C2", 416)]


def eight_seeds():
    """Q: is Branched CORO faster at the SAME batch size (T256 vs E4), and do the 3-seed results hold at 8 seeds?
    A seed that never reaches 0.9 counts as 1001 updates (worst rank)."""
    print("\n| config | episodes per update | seeds done | reached 0.9 | updates to 0.9 (per seed) | median | "
          "k rewarded episodes to 0.9 (median) | M turns to 0.9 (median) | rank test vs E4: updates, turns |")
    print("|---|---|---|---|---|---|---|---|---|")
    its, tus = {}, {}
    for label, pre, eps in EIGHT:
        rr = done_seeds(pre)
        if not rr:
            continue
        u = [first_at(r, "iter") for r in rr.values()]
        its[pre] = np.array([1001 if x is None else x for x in u])
        turns = [first_at(r, "env_steps") for r in rr.values()]
        tus[pre] = np.array([np.inf if x is None else x for x in turns])
        hit = [x for x in u if x is not None]
        p = "" if pre == "n_E4" or "n_E4" not in its else \
            f"p = {rank_test(its[pre], its['n_E4']):.3f}, {rank_test(tus[pre], tus['n_E4']):.3f}"
        print(f"| {label} | {eps} | {len(rr)} | {len(hit)} | {', '.join('-' if x is None else str(x) for x in u)} | "
              f"{np.median(its[pre]):.0f} | {np.median(its[pre]) * eps / 1e3:.0f} | "
              f"{np.median([1e12 if x is None else x for x in turns]) / 1e6:.2f} | {p} |")
    same = [s for s in (0, 1, 2) if (HERE / "runs" / f"n_E4_s{s}").exists()]
    for new, old in (("n_E4", "m_E4_coro_budget"), ("n_E5", "m_E5_tree_budget")):  # eval has its own RNG stream
        for s in same:
            a, b = done_seeds(new).get(s), read(f"{old}_s{s}")
            if a:
                print(f"bit-identical training loss {new}_s{s} vs {old}_s{s}: {[m['loss'] for m in a] == [m['loss'] for m in b]}")


def ppo_tuning():
    """The PPO-clip baselines had untuned advantages and clip. Seed 100 only (tuning seed), as for every other knob."""
    print("\n| advantages | clip | flat PPO-clip: final, updates to 0.9 | branched PPO-clip: final, updates to 0.9 |")
    print("|---|---|---|---|")
    for adv, lab in (("A0", "standardised"), ("A0dr", "centred only")):
        for c in (0.1, 0.2, 0.3):
            cells = []
            for o in ("ppo", "treeppo"):
                rr = done_seeds(f"p_{o}_{adv}_clip{c}", 101).get(100)
                cells.append("-" if rr is None else f"{evals(rr)[1][-3:].mean():.2f}, {first_at(rr, 'iter') or 'never'}")
            print(f"| {lab} | {c} | {cells[0]} | {cells[1]} |")
    print("\n| config (no drift budget, staleness <= 11) | final goal (seeds 0, 1, 2) | updates to 0.9 (per seed) |")
    print("|---|---|---|")
    for label, pre in (("E1 flat PPO-clip, default (standardised, clip 0.2)", "m_E1_ppo"),
                       ("flat PPO-clip, tuned (centred only, clip 0.2)", "p2_ppo_A0dr_clip0.2"),
                       ("E3 flat CORO", "m_E3_coro"),
                       ("B1 branched PPO-clip, default", "v_tree_ppo"),
                       ("branched PPO-clip, tuned (centred only, clip 0.2)", "p2_treeppo_A0dr_clip0.2"),
                       ("Branched CORO, beta 0.3", "f_tree_beta0.3")):
        rr = list(done_seeds(pre, 3).values())
        if rr:
            print(f"| {label} | {', '.join(f'{evals(r)[1][-3:].mean():.2f}' for r in rr)} | "
                  + ", ".join("-" if x is None else str(x) for x in (first_at(r, "iter") for r in rr)) + " |")


LOWPREC = [("PPO-clip", "ppo"), ("CORO", "coro"), ("Branched CORO", "tree"), ("ignore staleness", "none")]
LOWPREC_REF = {"q4": {"ppo": "m_E1_ppo", "coro": "m_E3_coro", "tree": "f_tree_beta0.3", "none": "m_E1b_naive"},
               "q4sev": {"ppo": "sev_q16_ppo", "coro": "sev_q16_coro", "tree": None, "none": "sev_q16_none"}}


def lowprec():
    """Sampler on 4-bit weights, trainer full precision (scripts/lowprec.sh). No budget; beta 0.3."""
    print("\n| objective | staleness <= 11: full precision | 4-bit sampler | staleness <= 31: full precision | 4-bit sampler |")
    print("|---|---|---|---|---|")
    stat = lambda rr, k: np.median([np.median([m[k] for m in r[100:]]) for r in rr])  # noqa: E731
    drift = {}
    for label, o in LOWPREC:
        cells = []
        for lvl in ("q4", "q4sev"):
            for pre in (LOWPREC_REF[lvl][o], f"{lvl}_{o}"):
                rr = list(done_seeds(pre, 3).values()) if pre else []
                cells.append(", ".join(f"{evals(r)[1][-3:].mean():.2f}" for r in rr) or "-")
                if rr:
                    drift[(o, pre)] = (stat(rr, "kl_k3"), stat(rr, "ess_traj"))
        print(f"| {label} | " + " | ".join(cells) + " |")
    print("\nmedian over runs of the per-run median (updates 100-999): per-token KL, trajectory ESS")
    for (o, pre), (kl, ess) in drift.items():
        print(f"  {pre}: KL {kl:.1e}, ESS {ess:.2f}")


ASYNC = [("CORO + drift budget", "a_coro_budget"), ("Branched CORO + budget", "a_tree_budget"), ("PPO-clip", "a_ppo")]


def async_runs():
    """Real asynchronous sampler and trainer (async_runner.py): staleness is measured, not set."""
    print("\n| async | final goal (per seed) | updates to 0.9 | M turns to 0.9 | minutes to 0.9 | staleness: first turn, last turn, max | trainer idle |")
    print("|---|---|---|---|---|---|---|")
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for i, (label, pre) in enumerate(ASYNC):
        rr = list(done_seeds(pre, 3).values())
        if not rr:
            continue
        done = 1000 if len(rr) == 3 else rr[0][-1]["iter"] + 1
        f = lambda xs, k=1, d=0: ", ".join("-" if x is None else f"{x / k:.{d}f}" for x in xs)  # noqa: E731
        st = lambda key: np.mean([np.mean([m[key] for m in r]) for r in rr])  # noqa: E731
        idle = np.mean([sum(m["wait"] for m in r) / r[-1]["wall"] for r in rr])
        print(f"| {label}{'' if len(rr) == 3 else f' ({len(rr)} of 3 runs done)'} | {f([evals(r)[1][-3:].mean() for r in rr], d=2)} | "
              f"{f([first_at(r, 'iter') for r in rr])} | {f([first_at(r, 'env_steps') for r in rr], 1e6, 2)} | "
              f"{f([first_at(r, 'wall') for r in rr], 60)} | {st('stal_first'):.1f}, {st('stal_last'):.1f}, "
              f"{max(m['staleness_max'] for r in rr for m in r)} | {idle:.0%} |")
        for r in rr:
            ev = [m for m in r if "eval_goal" in m]
            ax.plot([m["wall"] / 60 for m in ev], [m["eval_goal"] for m in ev], color=f"C{i}", lw=1, alpha=0.8, label=label if r is rr[0] else None)
    ax.set(xlabel="wall-clock minutes", ylabel="held-out goal rate", ylim=(0, 1.02), title="Real async sampler and trainer")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(HERE / "figs/async.png", dpi=150)


if __name__ == "__main__":
    (HERE / "figs").mkdir(exist_ok=True)
    r = load()
    table(r)
    curves(r)
    drift(r)
    severity()
    drift_table()
    controls()
    followup()
    e7()
    fixes()
    async_runs()
    eight_seeds()
    ppo_tuning()
    lowprec()
