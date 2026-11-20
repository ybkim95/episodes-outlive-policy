"""Regenerate figures from runs/*/metrics.jsonl only (no hand-edited numbers).

usage: python plots.py OUT.png RUN_GLOB [RUN_GLOB ...]
Runs whose names differ only in the trailing _s<seed> are one curve: mean line plus each seed faint.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = Path(__file__).parent / "runs"


def load(run_dir):
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    ev = [r for r in rows if "eval_return" in r]
    return [r["env_steps"] for r in ev], [r["eval_return"] for r in ev]


def main(out, *globs):
    groups = defaultdict(list)
    for g in globs:
        for d in sorted(RUNS.glob(g)):
            groups[re.sub(r"_s\d+$", "", d.name)].append(load(d))
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for i, (name, seeds) in enumerate(sorted(groups.items())):
        c = f"C{i}"
        for x, y in seeds:
            ax.plot(x, y, color=c, alpha=0.25, lw=1)
        n = min(len(x) for x, _ in seeds)  # seeds may have stopped at different points
        mean = [sum(y[j] for _, y in seeds) / len(seeds) for j in range(n)]
        ax.plot(seeds[0][0][:n], mean, color=c, lw=2, label=f"{name} (n={len(seeds)})")
    ax.axhline(1 / 8, color="gray", ls=":", lw=1, label="blind guess (1/8)")
    ax.set(xlabel="env steps (agent turns)", ylabel="held-out return", ylim=(0, 1.02))
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
