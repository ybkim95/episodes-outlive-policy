#!/bin/bash
# Two questions the talk left open. No code change: both are flag combinations of existing runs.
# X1: the fix the talk proposes for per-turn budgets. E7 (Branched CORO + per-bucket KL penalty, each bucket
#     with its own dual) but beta held at 0.3 (--beta_lr 0), so the anchor can no longer fall to its floor.
#     Same three penalty step sizes as scripts/e7.sh; the step is chosen on seed 100, all are reported.
# X2: is the branching gain specific to stale data? Branched CORO with no staleness (sync), to set
#     against E0 (flat CORO, sync).
cd "$(dirname "$0")/.." && mkdir -p logs
GPUS=(1 2 3); i=0
launch() { CUDA_VISIBLE_DEVICES=${GPUS[$((i % 3))]} nohup python3 runner.py --device cuda --name "$1" "${@:2}" > "logs/$1.log" 2>&1 & i=$((i + 1)); }
tree="--objective tree --groups 26 --group_size 4 --branch 4 --H 16 --W 16 --iters 1000 --eval_every 50"
for lr in 100 1000 10000; do
  for s in 100 0 1 2; do
    launch x_e7fix_lr${lr}_s$s $tree --U 4 --qmax 8 --seed $s --drift_budget 0.002 --beta_lr 0 --turn_penalty_lr $lr
  done
done
for s in 0 1 2; do
  launch x_tree_sync_s$s $tree --sync --seed $s
done
wait
