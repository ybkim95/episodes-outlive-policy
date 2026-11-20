"""Slide-sized figures for the talk (large type, one message each). Reads the same runs as analyze.py.

usage: python talk_figs.py   -> figs/talk_*.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analyze import E7_LRS, HERE, evals, final_goal, load, read, select_e7  # noqa: E402
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


def fig_turn_budget(runs):
    sel = select_e7()
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 6.6))
    curves = [("E5: no per-turn control", list(runs["E5_tree_budget"].values()), ORANGE, 4),
              ("E6: per-turn β", list(runs["E6_tree_turnbudget"].values()), PURPLE, 2.5)]
    curves += [(f"E7 ({lr})", [read(f"e7_lr{lr}_s{s}") for s in (0, 1, 2)], c, 2.5)
               for lr, c in zip(E7_LRS, (GOLD, "#8C6D1F", "#5E4A14"))]
    for lab, rows, col, lw in curves:
        x, m, _ = mean_curve(rows)
        axs[0].plot(x, m, color=col, lw=lw, label=lab)
    axs[0].set(xlabel="agent turns (millions)", ylabel="held-out goal rate", ylim=(0, 1.03))
    axs[0].legend(fontsize=15, loc="upper left", bbox_to_anchor=(-0.1, -0.17), ncol=2, frameon=False)
    kl = np.array([[np.nan if x is None else x for x in r["kl_by_stal"]] for r in read(f"e7_lr{sel}_s0")])
    early = np.nanmean(kl[:100], 0) * 1e3
    axs[1].bar(range(len(early)), early, color=NAVY)
    axs[1].set_xticks(range(0, len(early), 2))
    axs[1].axhline(2, color=ORANGE, lw=3, ls="--")
    axs[1].text(len(early) - 0.5, 2.08, "budget δ", ha="right", va="bottom", color=ORANGE, fontsize=16)
    axs[1].set(xlabel="staleness of the turn", ylabel="per-token KL ×10⁻³")
    save(fig, "talk_turn_budget.png")


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
    fig_branch()
