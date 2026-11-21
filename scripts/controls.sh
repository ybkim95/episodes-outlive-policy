#!/bin/bash
# Controls for the questions the main matrix left open. Seeds 0-2, same drift budget as E4/E5, one commit.
#   C1 tree_flat   E5's forked rollouts, plain CORO over the 16 leaves of a seed: is E5's speed the loss or the data?
#   C2 coro416     unforked CORO with E5's batch shape (26 seeds x 16 episodes = 416 per update, 1.6x the turns)
#   C3 e7_betaE5   E7's per-bucket penalty, with E5's beta path replayed: is E7 slow only because its beta floored early?
#   C4 e5_betaE7   E5 with E7's beta path replayed: does that beta path alone slow E5 down?
#   C5 sync_none   on-policy policy gradient (sync PPO is the same update: one step per batch keeps every ratio at 1)
#   R  E4, E5      rerun on this commit: same-code comparison, and a check against the original runs
cd "$(dirname "$0")/.." && mkdir -p logs
GPUS=(1 2 3); i=0
launch() { CUDA_VISIBLE_DEVICES=${GPUS[$((i % 3))]} nohup python3 runner.py --device cuda --name "$1" "${@:2}" > "logs/$1.log" 2>&1 & i=$((i + 1)); }
common="--H 16 --W 16 --iters 1000 --eval_every 50"
stale="--U 4 --qmax 8"
budget="--drift_budget 0.002 --beta_lr 1"
tree="--groups 26 --group_size 4 --branch 4"
for s in 0 1 2; do
  launch c_tree_flat_s$s --objective tree_flat $tree $stale $common $budget --seed $s
  launch c_coro416_s$s --objective coro --groups 26 --group_size 16 $stale $common $budget --seed $s
  launch c_e7_betaE5_s$s --objective tree $tree $stale $common $budget --turn_penalty_lr 100 --beta_path m_E5_tree_budget_s$s --seed $s
  launch c_e5_betaE7_s$s --objective tree $tree $stale $common $budget --beta_path e7_lr100_s$s --seed $s
  launch c_sync_none_s$s --objective none --sync $common --seed $s
  launch r_E4_coro_budget_s$s --objective coro $stale $common $budget --seed $s
  launch r_E5_tree_budget_s$s --objective tree $tree $stale $common $budget --seed $s
done
wait
