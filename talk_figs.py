"""Slide figures for the talk: one message each, drawn at the pixel size they take on a 1920x1080 slide.

usage: TALK_FONTS=<dir with Inter .ttf files> python talk_figs.py   -> figs/talk_*.png (transparent)
Without TALK_FONTS the figures fall back to DejaVu Sans.
"""
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from analyze import BETA_SWEEP, HERE, done_seeds, evals, final_goal, first_at, load, read  # noqa: E402
from runner import Config, staleness  # noqa: E402

for f in Path(os.environ.get("TALK_FONTS", "/nonexistent")).glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))

# the deck's palette, after the figures in humans&'s RL posts: warm greys for baselines, one teal for
# Branched CORO, rust for failures; axis titles in small spaced capitals
INK, MUTED, GRID, PAPER = "#161513", "#6B675F", "#E6E0D4", "#F9F6EF"
TEAL, RUST, DARK, GREY, LIGHT = "#00897B", "#B4492F", "#3B3D37", "#8A847A", "#C4BCAE"
TEAL_INK = "#176B64"  # teal for text
PX = 0.72  # points per slide pixel at dpi 100
plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"], "font.size": 22 * PX, "axes.titlesize": 24 * PX,
    "axes.labelsize": 22 * PX, "xtick.labelsize": 22 * PX, "ytick.labelsize": 22 * PX,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.edgecolor": LIGHT, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 1,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 10, "ytick.major.pad": 10,
    "axes.labelpad": 12, "axes.facecolor": "none", "figure.facecolor": "none", "savefig.transparent": True,
    "lines.solid_capstyle": "round", "lines.dash_capstyle": "round"})
OUT = HERE / "figs"


def fig(w, h, **kw):
    """A figure of w x h slide pixels."""
    return plt.subplots(figsize=(w / 100, h / 100), **kw)


def save(f, name, tight=True):
    if tight:
        f.tight_layout(pad=0.4)
    f.savefig(OUT / name, dpi=200)
    plt.close(f)


def label(ax, x, y, text, bold=False, **kw):
    kw.setdefault("color", INK)
    return ax.text(x, y, text, fontsize=24 * PX, fontweight=600 if bold else 400, va="center", **kw)


def save_layers(f, name, groups, tight=True):
    """Save the whole figure as name.png, then the same canvas split into layers a slide stacks and reveals one
    click at a time: name_base.png holds every artist outside the groups, name_<key>.png only that group."""
    if tight:
        f.tight_layout(pad=0.4)
    f.savefig(OUT / f"{name}.png", dpi=200)
    kids = [c for ax in f.axes for c in ax.get_children()]
    grouped = {a for arts in groups.values() for a in arts}
    assert grouped <= set(kids), "a layer artist is not a direct child of an axes"
    was = {c: c.get_visible() for c in kids}
    keeps = [("base", lambda c: c not in grouped)] + [(k, (lambda s: lambda c: c in s)(set(v))) for k, v in groups.items()]
    for key, keep in keeps:
        for c in kids:
            c.set_visible(was[c] and keep(c))
        f.savefig(OUT / f"{name}_{key}.png", dpi=200)
    plt.close(f)


def caps(text):
    """Spaced capitals for axis titles; Greek letters keep their case (an upper-case beta reads as B)."""
    return "\u200a".join(c.upper() if c.isascii() else c for c in text)


def titles(ax, y=None, x=None, **kw):
    """humans&-style axis titles: the y title sits horizontally above the axis, the x title below it."""
    style = dict(color=MUTED, fontsize=19 * PX, fontweight=600)
    if y:
        ax.text(0, 1.035, caps(y), transform=ax.transAxes, va="bottom", ha="left", **style, **kw)
    if x:
        ax.set_xlabel(caps(x), **style)


def spread(ys, gap, top):
    """Label heights at least gap apart, in the data's order, none above top, so direct labels never overlap."""
    order = np.argsort(ys)[::-1]
    out = np.minimum(np.array(ys, float), top)
    for a, b in zip(order, order[1:]):
        out[b] = min(out[b], out[a] - gap)
    return out


def by_update(names):
    rows = [read(n) for n in names]
    it = [r["iter"] + 1 for r in rows[0] if "eval_goal" in r]
    ys = [evals(r)[1] for r in rows]
    n = min(len(y) for y in ys)
    return np.array(it[:n]), np.mean([y[:n] for y in ys], 0), [y[:n] for y in ys]


def fig_staleness():
    """Per-turn staleness for 24 episodes of one batch: older at the start, heavy-tailed across episodes."""
    s = staleness(Config(H=16, W=16, U=4, qmax=8), np.random.default_rng(3), 24)
    s = s[np.argsort(s[:, 0])]
    cmap = LinearSegmentedColormap.from_list("age", ["#E3EFED", "#8CC3BA", TEAL, "#0F4A44"])
    f, ax = fig(980, 620)
    m = ax.pcolormesh(s, cmap=cmap, vmin=0, vmax=s.max(), edgecolors=PAPER, linewidth=2.5)
    ax.invert_yaxis()
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    ax.set(xticks=np.array([0, 4, 8, 12, 15]) + 0.5, xticklabels=[0, 4, 8, 12, 15], yticks=[])
    titles(ax, "24 episodes from one batch", "turn within the episode")
    cb = f.colorbar(m, ax=ax, pad=0.02, fraction=0.04, aspect=30)
    cb.outline.set_visible(False)
    cb.ax.tick_params(size=0, colors=MUTED)
    cb.set_label(caps("trainer updates since this turn was sampled"), color=MUTED, fontsize=19 * PX, fontweight=600)
    save(f, "talk_staleness.png")


def fig_main(runs):
    """Goal rate against sampled agent turns. The two budget runs are the 8-run reruns (n_E4, n_E5; eval every 10
    updates), the others 3 runs. Each curve is the mean goal rate at each eval, plotted at the mean turns sampled by
    then. The arrow joins where the two mean curves cross 0.9; the slide quotes the median of per-run crossings."""
    f, ax = fig(1120, 640)
    spec = [("E5_tree_budget", "Branched CORO + drift budget", TEAL, "-", 4.5),
            ("E4_coro_budget", "CORO + drift budget", DARK, "-", 3),
            ("E1b_naive", "ignore staleness (3 runs)", RUST, (0, (2, 3)), 2.5)]  # 3-run curves next to 8-run ones mislead
    cross, lines, curve = {}, {}, {}
    eight = {"E5_tree_budget": "n_E5", "E4_coro_budget": "n_E4"}
    for cfg, lab, col, ls, lw in spec:
        rr = list(done_seeds(eight[cfg]).values()) if cfg in eight else list(runs[cfg].values())
        xs, ys = zip(*(evals(r) for r in rr))
        n = min(len(y) for y in ys)
        x, m = np.mean([v[:n] for v in xs], 0) / 1e6, np.mean([y[:n] for y in ys], 0)
        (lines[cfg],) = ax.plot(x, m, color=col, ls=ls, lw=lw, label=lab)
        curve[cfg] = x, m
        i = np.argmax(m >= 0.9)
        cross[cfg] = x[i - 1] + (0.9 - m[i - 1]) / (m[i] - m[i - 1]) * (x[i] - x[i - 1]) if m.max() >= 0.9 else None
    a, b = cross["E5_tree_budget"], cross["E4_coro_budget"]
    arrow = ax.annotate("", xy=(b, 0.9), xytext=(a, 0.9), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.8,
                                                                           mutation_scale=18, shrinkA=6, shrinkB=6))
    gain = label(ax, (a + b) / 2, 0.86, "2×", ha="center", bold=True)
    x5, m5 = curve["E5_tree_budget"]
    i = np.argmax(m5 >= 0.62)
    xl = x5[i - 1] + (0.62 - m5[i - 1]) / (m5[i] - m5[i - 1]) * (x5[i] - x5[i - 1])
    name5 = label(ax, xl - 0.07, 0.62, "Branched CORO\n+ drift budget", ha="right", bold=True, color=TEAL_INK)
    ax.text(0.03, 0.925, "0.9", color=MUTED, fontsize=20 * PX)
    ax.axhline(0.9, color=MUTED, lw=1, ls=(0, (1, 3)))
    ax.axhline(1 / 8, color=MUTED, lw=1, ls=(0, (1, 3)))
    ax.text(4.12, 0.155, "chance", color=MUTED, ha="right", fontsize=20 * PX)
    ax.legend(handles=[lines[k] for k in ("E4_coro_budget", "E1b_naive")], loc="center right",
              bbox_to_anchor=(1.0, 0.56), frameon=False, labelcolor=INK, fontsize=24 * PX, handlelength=1.8,
              borderaxespad=0)
    ax.set(ylim=(0, 1.04), xlim=(0, 4.15), yticks=[0, 0.25, 0.5, 0.75, 1])
    titles(ax, "held-out goal rate, mean of 8 runs", "agent turns sampled (millions)")
    save_layers(f, "talk_main", {"teal": [lines["E5_tree_budget"], name5], "gain": [arrow, gain]})


CONTROL_ROWS = [  # (label, run prefix, family); "n_" prefixes are the 8-run reruns (eval every 10 updates)
    ("Branched CORO, 416 episodes", "n_E5", "tree"),
    ("Branched CORO, 256 episodes", "n_T256", "tree"),
    ("level 2 only (416)", "v_tree_lvl2_bp", "tree"),
    ("siblings redraw the goal change (416)", "v_tree_resample_bp", "tree"),
    ("same branches, PPO-clip, tuned (416)", "p2_treeppo_A0dr_clip0.2", "flat"),
    ("same forked data, flat CORO (416)", "n_C1", "flat"),
    ("independent episodes, flat CORO (416)", "n_C2", "flat"),
    ("independent episodes, flat CORO (256)", "n_E4", "flat"),
]


def control_runs(pre):
    if pre.startswith("n_"):
        return list(done_seeds(pre).values())
    return [read(f"{pre}_s{s}") for s in (0, 1, 2)]


def fig_controls():
    """Updates to 0.9 per run (dots) and the median (bar), for Branched CORO, its ablations and the flat controls.
    A run that never reaches 0.9 is an open dot at "never" and counts as the slowest in the median."""
    f, ax = fig(1664, 600)
    f.subplots_adjust(left=0.37, right=0.89, top=0.97, bottom=0.16)
    never, tree = 860, []
    for i, (lab, pre, fam) in enumerate(CONTROL_ROWS):
        arts = tree if fam == "tree" else []
        col = TEAL if fam == "tree" else GREY
        its = [first_at(r, "iter") for r in control_runs(pre)]
        seen = {}
        for x in its:
            v = never if x is None else x
            k = seen.setdefault(v, 0)
            seen[v] += 1
            dy = 0 if its.count(x) == 1 else (k - (its.count(x) - 1) / 2) * 0.16
            arts += ax.plot(v, i + dy, "o", ms=10, mew=2, mec=PAPER if x else col, color=col if x else "none", alpha=0.9, zorder=4)
        med = np.median([10_000 if x is None else x for x in its])
        hit = sum(x is not None for x in its)
        if med < 10_000:
            arts += ax.plot([med] * 2, [i - 0.34, i + 0.34], color=INK, lw=3, zorder=5)
        txt = f"{med:.0f}" if med < 10_000 else "–"
        txt += "" if hit == len(its) else f"  ({hit} of {len(its)})"
        arts.append(ax.text(1.005, i, txt, transform=ax.get_yaxis_transform(), color=INK, va="center",
                            fontsize=24 * PX, fontweight=600 if fam == "tree" else 400))
        arts.append(ax.text(-0.02, i, f"{lab}  ·  {len(its)}", transform=ax.get_yaxis_transform(), color=INK, va="center",
                            ha="right", fontsize=24 * PX))
    ax.axhline(3.5, color=GRID, lw=1.5)
    ax.text(never, len(CONTROL_ROWS) - 0.2, "never", color=MUTED, ha="center", va="center", fontsize=20 * PX)
    ax.set(ylim=(len(CONTROL_ROWS) - 0.4, -0.6), yticks=[], xlim=(240, 880), xticks=[300, 400, 500, 600, 700, 800])
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.spines["bottom"].set_visible(False)
    titles(ax, x="trainer updates to reach a 0.9 goal rate  ·  dots: runs, bar: median")
    save_layers(f, "talk_controls", {"teal": tree}, tight=False)


def fig_severity():
    names = [{"ppo": "m_E1_ppo", "oapl": "m_E2_oapl", "coro": "m_E3_coro", "none": "m_E1b_naive"}] + \
            [{o: f"sev_q{q}_{o}" for o in ("ppo", "oapl", "coro", "none")} for q in (16, 48)]
    f, ax = fig(1000, 620)
    f.subplots_adjust(left=0.11, right=0.74, top=0.9, bottom=0.15)
    ends, arts = [], {}
    for obj, lab, col, mk, ls in [("ppo", "PPO-clip", DARK, "o", "-"), ("oapl", "OAPL", GREY, "s", "-"),
                                  ("coro", "CORO", LIGHT, "D", "-"), ("none", "ignore staleness", RUST, "o", (0, (2, 3)))]:
        v = [[final_goal(n[obj], s) for s in (0, 1, 2)] for n in names]
        means = [np.mean(vs) for vs in v]
        arts[obj] = ax.plot(range(3), means, color=col, lw=3, ls=ls, marker=mk, ms=11, mec=PAPER, mew=2)
        ends.append((means[-1], lab, obj))
    ys = spread([e[0] for e in ends], 0.07, 1.0)
    for (_, lab, obj), y in zip(ends, ys):
        arts[obj].append(label(ax, 2.12, y, lab))
    ax.axhline(1 / 8, color=MUTED, lw=1, ls=(0, (1, 3)))
    ax.text(0, 0.155, "chance", color=MUTED, fontsize=20 * PX)
    ax.set_xticks(range(3), ["11", "31", "63"])
    ax.set(ylim=(0, 1.05), xlim=(-0.15, 2.1))
    titles(ax, "final goal rate, mean of 3 runs", "maximum staleness within an episode (trainer updates)")
    save_layers(f, "talk_severity", {"fail": arts["none"]}, tight=False)


def fig_turn_budget():
    """Per-turn budgets against Branched CORO, and why the KL-penalty version fails: its global beta reaches
    the floor while it is still learning. Holding beta fixed (X1, layer "fix") repairs it."""
    f = plt.figure(figsize=(16.64, 5.6))
    a0 = f.add_axes([0.04, 0.15, 0.36, 0.74])
    a1 = f.add_axes([0.67, 0.15, 0.31, 0.74])
    curves = [("Branched CORO + budget", "r_E5_tree_budget", TEAL, "-", 4.5),
              ("same penalty, β fixed", "x_e7fix_lr100", DARK, (0, (4, 3)), 3),
              ("per-turn KL penalty", "e7_lr100", RUST, "-", 3),
              ("per-turn β", "m_E6_tree_turnbudget", RUST, (0, (2, 3)), 3)]
    ends, fix = [], []
    for lab, pre, col, ls, lw in curves:
        x, m, _ = by_update([f"{pre}_s{s}" for s in (0, 1, 2)])
        ln, = a0.plot(x, m, color=col, ls=ls, lw=lw)
        fix += [ln] if pre.startswith("x_") else []
        ends.append((m[-1], lab, np.mean([final_goal(pre, s) for s in (0, 1, 2)])))
    for (_, lab, fin), y in zip(ends, spread([e[0] + 1e-3 * (k == 0) for k, e in enumerate(ends)], 0.075, 1.0)):  # ties: Branched on top
        t = label(a0, 1025, y, f"{lab}  {fin:.2f}", bold=lab.startswith("Branched"))
        fix += [t] if "fixed" in lab else []
    a0.text(1025, 1.09, "final", color=MUTED, fontsize=20 * PX)
    a0.set(ylim=(0, 1.04), xlim=(0, 1000), yticks=[0, 0.25, 0.5, 0.75, 1])
    titles(a0, "held-out goal rate, mean of 3 runs", "trainer updates")
    for pre, col, lab, xy, ha in (("m_E5_tree_budget", TEAL, "Branched CORO", (650, 0.04), "left"),
                                  ("e7_lr100", RUST, "per-turn\nKL penalty", (390, 5e-3), "right")):
        b = np.mean([[r["beta"] for r in read(f"{pre}_s{s}")] for s in (0, 1, 2)], 0)
        a1.plot(np.arange(1, len(b) + 1), b, color=col, lw=3.5 if col == TEAL else 3)
        label(a1, *xy, lab, ha=ha)
    a1.axhline(1e-3, color=MUTED, lw=1, ls=(0, (1, 3)))
    a1.text(30, 1.25e-3, "floor", color=MUTED, fontsize=20 * PX)
    a1.set(yscale="log", xlim=(0, 1000), ylim=(7e-4, 0.6))
    titles(a1, "global β, log scale", "trainer updates")
    save_layers(f, "talk_turn_budget", {"beta": list(a1.get_children()), "fix": fix}, tight=False)


def fig_async():
    """Real async sampler and trainer (async_runner.py): goal rate against trainer updates, each run thin and the
    mean of the finished runs thick. Branched CORO is the "teal" layer."""
    f, ax = fig(1100, 600)
    f.subplots_adjust(left=0.08, right=0.62, top=0.9, bottom=0.15)
    teal, ends = [], []
    for lab, pre, col, lw in (("Branched CORO + budget", "a_tree_budget", TEAL, 4.5), ("CORO + budget", "a_coro_budget", DARK, 3),
                              ("PPO-clip", "a_ppo", GREY, 3)):
        rr = list(done_seeds(pre, 3).values())
        if not rr:
            continue
        arts = teal if col == TEAL else []
        ys = [evals(r)[1] for r in rr]
        x = np.array([m["iter"] + 1 for m in rr[0] if "eval_goal" in m])
        for y in ys:
            arts += ax.plot(x, y, color=col, lw=1.2, alpha=0.45)
        m = np.mean(ys, 0)
        arts += ax.plot(x, m, color=col, lw=lw)
        ends.append((m[-1], f"{lab}  {np.mean([y[-3:].mean() for y in ys]):.2f}", arts, col == TEAL))
    for (_, txt, arts, bold), y in zip(ends, spread([e[0] for e in ends], 0.075, 1.0)):
        arts.append(label(ax, 1025, y, txt, bold=bold))
    ax.text(1025, 1.09, "final, mean of runs", color=MUTED, fontsize=20 * PX)
    ax.axhline(1 / 8, color=MUTED, lw=1, ls=(0, (1, 3)))
    ax.set(ylim=(0, 1.04), xlim=(0, 1000), yticks=[0, 0.25, 0.5, 0.75, 1])
    titles(ax, "held-out goal rate · thin: each run", "trainer updates")
    save_layers(f, "talk_async", {"teal": teal}, tight=False)


def fig_beta_sweep():
    """Fixed beta from the first update, no drift budget: too small never learns, too large is slow."""
    f, ax = fig(1100, 620)
    f.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.15)
    for (obj, runs), col, off in zip(BETA_SWEEP.items(), (DARK, TEAL), (0.92, 1.09)):
        bs, means = sorted(runs), []
        for b in bs:
            its = [first_at(read(f"{runs[b]}_s{s}"), "iter") or 1000 for s in (0, 1, 2)]
            ax.plot([b * off] * 3, its, "o", color=col, ms=9, alpha=0.35, mew=0)
            means.append(np.mean(its))
        ax.plot(bs, means, color=col, lw=4.5 if col == TEAL else 3, marker="o", ms=11, mec=PAPER, mew=2)
        label(ax, 4.2, means[-1], obj, bold=col == TEAL)
    ax.axhline(1000, color=MUTED, lw=1, ls=(0, (1, 3)))
    ax.text(1.4e-3, 1035, "never reached 0.9 (drawn at 1000)", color=MUTED, fontsize=20 * PX)
    ax.set(xscale="log", ylim=(250, 1080), xlim=(7e-4, 30), xticks=[1e-3, 0.03, 0.3, 3], xticklabels=["0.001", "0.03", "0.3", "3"])
    titles(ax, "trainer updates to reach 0.9", "β, fixed from the first update")
    save(f, "talk_beta_sweep.png", tight=False)


def fig_branch():
    """Schematic: one trunk, forked at turn b into k continuations from the same state."""
    f, ax = fig(1000, 520)
    ax.axis("off")
    b, H, k = 6, 16, 4
    ax.plot(range(b + 1), [0] * (b + 1), color=INK, lw=4, marker="o", ms=10, mec=PAPER, mew=2, zorder=3)
    for e, ok in zip(np.linspace(1.6, -1.6, k), (True, False, True, False)):
        xs = np.arange(b, H)
        ys = e * (1 - np.exp(-(xs - b) / 1.6))
        ax.plot(xs, ys, color=TEAL, lw=3, marker="o", ms=8, mec=PAPER, mew=1.5, zorder=2)
        label(ax, H - 0.35, ys[-1], "r = 1" if ok else "r = 0", bold=ok)
    ax.plot(b, 0, "o", ms=18, color=PAPER, mec=INK, mew=3, zorder=4)
    label(ax, b - 0.3, -0.55, "fork at turn b", ha="right")
    for x0, x1, txt in ((0, b, "shared prefix, sampled once"), (b, H - 1, "k = 4 continuations of the same state")):
        ax.annotate("", xy=(x1, -2.25), xytext=(x0, -2.25), arrowprops=dict(arrowstyle="|-|", color=MUTED, lw=1.5,
                                                                            mutation_scale=6))
        ax.text((x0 + x1) / 2, -2.65, txt, ha="center", color=MUTED, fontsize=22 * PX, va="center")
    ax.set(xlim=(-0.5, H + 1), ylim=(-2.9, 2.0))
    save(f, "talk_branch.png")


if __name__ == "__main__":
    r = load()
    fig_staleness()
    fig_main(r)
    fig_severity()
    fig_turn_budget()
    fig_async()
    fig_controls()
    fig_beta_sweep()
    fig_branch()
