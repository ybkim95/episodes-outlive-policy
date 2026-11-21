#!/bin/bash
# Two questions the controls left open. Seeds 0-2, staleness as in the main matrix.
#   B  fixed beta from the start, no drift budget. Control C4 only changed beta by 2-3x before Branched
#      CORO reached 0.9 (E7's beta hit the floor after that), so it could not show that beta is inert.
#      beta = 0.001 also switches the anchor term off: is CORO then just policy gradient that ignores staleness (E1b)?
#      Flat CORO at beta = 0.3 is E3 in main.sh.
#   H  32-turn episodes: E4, C1 and E5 as before. The earlier 32-turn runs were synchronous and learned nothing.
cd "$(dirname "$0")/.." && mkdir -p logs
GPUS=(1 2 3); i=0
launch() { CUDA_VISIBLE_DEVICES=${GPUS[$((i % 3))]} nohup python3 runner.py --device cuda --name "$1" "${@:2}" > "logs/$1.log" 2>&1 & i=$((i + 1)); }
common="--iters 1000 --eval_every 50"
stale="--U 4 --qmax 8"
budget="--drift_budget 0.002 --beta_lr 1"
tree="--groups 26 --group_size 4 --branch 4"
for s in 0 1 2; do
  for b in 0.001 0.03 3; do
    launch f_coro_beta${b}_s$s --objective coro --beta $b --H 16 --W 16 $stale $common --seed $s
  done
  for b in 0.001 0.03 0.3 3; do
    launch f_tree_beta${b}_s$s --objective tree $tree --beta $b --H 16 --W 16 $stale $common --seed $s
  done
  launch h32_E4_coro_budget_s$s --objective coro --H 32 --W 32 $stale $common $budget --seed $s
  launch h32_C1_tree_flat_s$s --objective tree_flat $tree --H 32 --W 32 $stale $common $budget --seed $s
  launch h32_E5_tree_budget_s$s --objective tree $tree --H 32 --W 32 $stale $common $budget --seed $s
done
wait
