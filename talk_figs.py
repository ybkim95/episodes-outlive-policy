"""Slide-sized figures for the talk (large type, one message each). Reads the same runs as analyze.py.

usage: python talk_figs.py   -> figs/talk_*.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analyze import HERE, evals, final_goal, load, read  # noqa: E402
from runner import Config, staleness  # noqa: E402

plt.rcParams.update({"font.size": 18, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#3D4757", "axes.labelcolor": "#3D4757", "xtick.color": "#3D4757",
                     "ytick.color": "#3D4757", "font.family": "DejaVu Sans", "figure.facecolor": "#F7F6F2",
                     "axes.facecolor": "#F7F6F2", "savefig.facecolor": "#F7F6F2"})
NAVY, BLUE, ORANGE, GREY, RED, PURPLE, GOLD = "#14213D", "#2F6DB5", "#E07A1F", "#8A94A6", "#B23A48", "#7B5EA7", "#B8891A"
OUT = HERE / "figs"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


def mean_curve(rows_by_seed):
    xs, ys = zip(*(evals(r) for r in rows_by_seed))
    n = min(len(y) for y in ys)
    return xs[0][:n] / 1e6, np.mean([y[:n] for y in ys], 0), [y[:n] for y in ys]


def fig_staleness():
    """Per-turn staleness for 24 episodes of one batch: older at the start, heavy-tailed across episodes."""
    s = staleness(Config(H=16, W=16, U=4, qmax=8), np.random.default_rng(3), 24)
    s = s[np.argsort(s[:, 0])]
    fig, ax = plt.subplots(figsize=(8, 5.4))
    im = ax.imshow(s, cmap="cividis_r", aspect="auto", vmin=0, vmax=s.max())
    ax.set(xlabel="turn within the episode", ylabel="episode (sorted)", xticks=[0, 4, 8, 12, 15], yticks=[])
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("staleness (trainer updates)")
    save(fig, "talk_staleness.png")


def fig_main(runs):
    """Main matrix at matched agent turns: the four corrected objectives and the uncorrected one."""
    fig, ax = plt.subplots(figsize=(11, 6.2))
    spec = [("E1b_naive", "ignore staleness", RED, "--", 2), ("E1_ppo", "PPO-clip (token IS)", BLUE, "-", 2.5),
            ("E3_coro", "CORO", NAVY, "-", 2.5), ("E4_coro_budget", "CORO + drift budget", GREY, "-", 2.5),
            ("E5_tree_budget", "Branched CORO + budget", ORANGE, "-", 4)]
    for cfg, lab, col, ls, lw in spec:
        x, m, ys = mean_curve(list(runs[cfg].values()))
        for y in ys:
            ax.plot(x, y, color=col, alpha=0.18, lw=1)
        ax.plot(x, m, color=col, ls=ls, lw=lw, label=lab)
    ax.axhline(0.9, color="#3D4757", lw=1, ls=":")
    ax.axhline(1 / 8, color="#3D4757", lw=1, ls=":")
    ax.text(4.15, 0.125, "chance", va="center", fontsize=15, color="#3D4757")
    ax.set(xlabel="agent turns generated (millions)", ylabel="held-out goal rate", ylim=(0, 1.03), xlim=(0, 4.15))
    ax.legend(fontsize=15, loc="center right", frameon=False)
    save(fig, "talk_main.png")


def fig_severity():
    names = [{"ppo": "m_E1_ppo", "oapl": "m_E2_oapl", "coro": "m_E3_coro", "none": "m_E1b_naive"}] + \
            [{o: f"sev_q{q}_{o}" for o in ("ppo", "oapl", "coro", "none")} for q in (16, 48)]
    fig, ax = plt.subplots(figsize=(10, 6.2))
    for i, (obj, lab, col, mk) in enumerate([("ppo", "PPO-clip", BLUE, "o"), ("oapl", "OAPL", GREY, "s"),
                                             ("coro", "CORO", NAVY, "D"), ("none", "ignore staleness", RED, "v")]):
        v = [[final_goal(n[obj], s) for s in (0, 1, 2)] for n in names]
        for j, vs in enumerate(v):
            ax.scatter([j + 0.06 * (i - 1.5)] * 3, vs, color=col, alpha=0.35, s=40, marker=mk)
        ax.plot(range(3), [np.mean(vs) for vs in v], color=col, lw=3, marker=mk, ms=10, label=lab)
    ax.axhline(1 / 8, color="#3D4757", lw=1, ls=":")
    ax.set_xticks(range(3), ["max 11", "max 31", "max 63"])
    ax.set(xlabel="staleness within an episode (trainer updates)", ylabel="final goal rate (3 seeds)", ylim=(0, 1.05))
    ax.legend(fontsize=15, loc="center left", frameon=False)
    save(fig, "talk_severity.png")


def by_update(names):
    rows = [read(n) for n in names]
    it = [r["iter"] + 1 for r in rows[0] if "eval_goal" in r]
    ys = [evals(r)[1] for r in rows]
    n = min(len(y) for y in ys)
    return np.array(it[:n]), np.mean([y[:n] for y in ys], 0), [y[:n] for y in ys]


def fig_controls():
    """Per update: more episodes (C2), the same forked data with a flat loss (C1), the two-level loss (E5)."""
    fig, ax = plt.subplots(figsize=(11, 6.2))
    spec = [("r_E4_coro_budget", "CORO, 256 episodes / update", GREY, 2.5), ("c_coro416", "CORO, 416 episodes / update", NAVY, 2.5),
            ("c_tree_flat", "forked data (416 leaves), flat CORO", BLUE, 2.5), ("r_E5_tree_budget", "forked data, Branched CORO", ORANGE, 4)]
    for pre, lab, col, lw in spec:
        x, m, ys = by_update([f"{pre}_s{s}" for s in (0, 1, 2)])
        for y in ys:
            ax.plot(x, y, color=col, alpha=0.18, lw=1)
        ax.plot(x, m, color=col, lw=lw, label=lab)
    ax.axhline(0.9, color="#3D4757", lw=1, ls=":")
    ax.set(xlabel="trainer updates", ylabel="held-out goal rate", ylim=(0, 1.03), xlim=(0, 800))
    ax.legend(fontsize=15, loc="lower right", frameon=False)
    save(fig, "talk_controls.png")


def fig_turn_budget(runs):
    """Per-turn budgets against Branched CORO, and why the KL-penalty version fails: its global beta reaches
    the floor while it is still learning. Replaying Branched CORO's beta path fixes it."""
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 6.6))
    curves = [("Branched CORO + budget", "r_E5_tree_budget", ORANGE, "-", 4),
              ("+ per-turn β", "m_E6_tree_turnbudget", PURPLE, "-", 2.5),
              ("+ per-turn KL penalty", "e7_lr100", GOLD, "-", 3),
              ("+ per-turn KL penalty, β path of\n   Branched CORO replayed", "c_e7_betaE5", "#5E4A14", "--", 2.5)]
    for lab, pre, col, ls, lw in curves:
        x, m, _ = by_update([f"{pre}_s{s}" for s in (0, 1, 2)])
        axs[0].plot(x, m, color=col, ls=ls, lw=lw, label=lab)
    axs[0].set(xlabel="trainer updates", ylabel="held-out goal rate", ylim=(0, 1.03))
    axs[0].legend(fontsize=14, loc="upper left", bbox_to_anchor=(-0.1, -0.17), ncol=2, frameon=False)
    for lab, pre, col in (("Branched CORO + budget", "m_E5_tree_budget", ORANGE), ("+ per-turn KL penalty", "e7_lr100", GOLD)):
        b = np.mean([[r["beta"] for r in read(f"{pre}_s{s}")] for s in (0, 1, 2)], 0)
        axs[1].plot(np.arange(1, len(b) + 1), b, color=col, lw=3, label=lab)
    axs[1].axhline(1e-3, color="#3D4757", lw=1, ls=":")
    axs[1].text(1000, 1.15e-3, "floor", ha="right", va="bottom", fontsize=15, color="#3D4757")
    axs[1].set(xlabel="trainer updates", ylabel="global β (drift-budget dual)", yscale="log")
    axs[1].legend(fontsize=14, frameon=False)
    save(fig, "talk_turn_budget.png")


def fig_beta_sweep():
    """Fixed beta from the first update, no drift budget: too small never learns, too large is slow."""
    from analyze import BETA_SWEEP, first_at
    fig, ax = plt.subplots(figsize=(11, 6.2))
    for (obj, runs), col, off in zip(BETA_SWEEP.items(), (BLUE, ORANGE), (0.93, 1.08)):
        bs, means = sorted(runs), []
        for b in bs:
            its = [first_at(read(f"{runs[b]}_s{s}"), "iter") or 1000 for s in (0, 1, 2)]
            ax.scatter([b * off] * 3, its, color=col, s=40, alpha=0.45)
            means.append(np.mean(its))
        ax.plot(bs, means, color=col, lw=4 if obj.startswith("Branched") else 2.5, marker="o", ms=10, label=obj)
    ax.axhline(1000, color="#3D4757", lw=1, ls=":")
    ax.text(1.6e-3, 985, "never reached 0.9", fontsize=15, color="#3D4757", va="top")
    ax.set(xscale="log", xlabel="β, fixed from the first update", ylabel="updates to reach 0.9", ylim=(250, 1040))
    ax.legend(fontsize=15, loc="lower left", frameon=False)
    save(fig, "talk_beta_sweep.png")


def fig_branch():
    """Schematic: one trunk, forked at turn b into k continuations."""
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.axis("off")
    b, H, k = 6, 16, 4
    ax.plot(range(b + 1), [0] * (b + 1), color=NAVY, lw=4, marker="o", ms=9)
    ends = np.linspace(1.5, -1.5, k)
    rng = np.random.default_rng(1)
    for e, ok in zip(ends, (True, False, True, False)):
        xs = np.arange(b, H)
        ys = np.linspace(0, e, len(xs)) + np.r_[0, rng.normal(0, 0.06, len(xs) - 1)]
        ax.plot(xs, ys, color=ORANGE, lw=2.5, marker="o", ms=6)
        ax.text(H - 0.4, e, "r = 1" if ok else "r = 0", va="center", fontsize=17, color=NAVY)
    ax.annotate("", xy=(b, -2.15), xytext=(0, -2.15), arrowprops=dict(arrowstyle="<->", color=NAVY, lw=2))
    ax.text(b / 2, -2.5, "shared prefix (generated once)", ha="center", fontsize=16, color=NAVY)
    ax.annotate("", xy=(H - 1, -2.15), xytext=(b, -2.15), arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=2))
    ax.text((b + H - 1) / 2, -2.5, f"k = {k} continuations from the same state", ha="center", fontsize=16, color=ORANGE)
    ax.text(b, 0.35, "fork at turn b", ha="center", fontsize=16, color=NAVY)
    ax.set(xlim=(-0.5, H + 0.8), ylim=(-2.8, 1.9))
    save(fig, "talk_branch.png")


if __name__ == "__main__":
    r = load()
    fig_staleness()
    fig_main(r)
    fig_severity()
    fig_turn_budget(r)
    fig_controls()
    fig_beta_sweep()
    fig_branch()
