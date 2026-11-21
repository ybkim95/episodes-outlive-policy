#!/bin/bash
# Where does Branched CORO's speed come from? Seeds 0-2, same staleness and drift budget as E4/E5.
#   M1 tree_lvl2      E5 without level 1: sibling contrast only (prefix turns get no gradient)
#   M2 tree_resample  E5, but forked siblings redraw the pending goal shift: shared history, independent user future
#   M3 coro_crn       E4, but the 8 episodes of a group share one goal-shift schedule, as forked siblings do
cd "$(dirname "$0")/.." && mkdir -p logs
GPUS=(1 2 3); i=0
launch() { CUDA_VISIBLE_DEVICES=${GPUS[$((i % 3))]} nohup python3 runner.py --device cuda --name "$1" "${@:2}" > "logs/$1.log" 2>&1 & i=$((i + 1)); }
common="--H 16 --W 16 --iters 1000 --eval_every 50"
stale="--U 4 --qmax 8"
budget="--drift_budget 0.002 --beta_lr 1"
tree="--groups 26 --group_size 4 --branch 4"
for s in 0 1 2; do
  launch k_tree_lvl2_s$s --objective tree $tree $stale $common $budget --tree_no_level1 --seed $s
  launch k_tree_resample_s$s --objective tree $tree $stale $common $budget --fork_resample --seed $s
  launch k_coro_crn_s$s --objective coro $stale $common $budget --crn --seed $s
done
wait
