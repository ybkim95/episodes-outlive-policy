#!/bin/bash
# Main matrix, H=16 / W=16 (full context), GPUs 1-3. Report seeds 0-2; tune on seed 100 only.
# Staleness: sampler weights advance every U=4 turns inside an episode, plus a heavy-tailed per-episode
# queue delay q <= 8, so one episode spans up to 4 versions and staleness ranges 0..11.
#   usage: main.sh a            -> base configs (seeds 0-2) + drift-budget sweep (seed 100)
#          main.sh b DELTA LR   -> drift-budget configs (seeds 0-2)
cd "$(dirname "$0")/.." && mkdir -p logs
GPUS=(1 2 3); i=0
launch() { CUDA_VISIBLE_DEVICES=${GPUS[$((i % 3))]} nohup python3 runner.py --device cuda --name "$1" "${@:2}" > "logs/$1.log" 2>&1 & i=$((i + 1)); }
common="--H 16 --W 16 --iters 1000 --eval_every 50"
stale="--U 4 --qmax 8"
tree="--objective tree --groups 26 --group_size 4 --branch 4"
if [ "$1" = a ]; then
  for s in 0 1 2; do
    launch m_E0_coro_sync_s$s --objective coro --sync $common --seed $s
    launch m_E1_ppo_s$s --objective ppo $stale $common --seed $s
    launch m_E1b_naive_s$s --objective none $stale $common --seed $s
    launch m_E2_oapl_s$s --objective oapl $stale $common --seed $s
    launch m_E3_coro_s$s --objective coro $stale $common --seed $s
  done
  for d in 0.002 0.01; do for lr in 1 10; do
    launch tune_coro_budget_d${d}_lr${lr}_s100 --objective coro $stale $common --seed 100 --drift_budget $d --beta_lr $lr
  done; done
else
  d=$2; lr=$3
  for s in 0 1 2; do
    launch m_E4_coro_budget_s$s --objective coro $stale $common --seed $s --drift_budget $d --beta_lr $lr
    launch m_E5_tree_budget_s$s $tree $stale $common --seed $s --drift_budget $d --beta_lr $lr
    launch m_E6_tree_turnbudget_s$s $tree $stale $common --seed $s --drift_budget $d --beta_lr $lr --drift_per_turn
  done
fi
wait
