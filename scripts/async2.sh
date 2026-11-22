#!/bin/bash
# The rest of async.sh on GPUs 1-2 (GPU 3 filled up, and a_ppo_s0 failed at start with CUDA out of memory).
# Waits for the seed-0 wave, then runs seeds 1 and 2 (three methods at a time, as before) and last a_ppo_s0
# alone, so its wall-clock and staleness are not comparable with the other seed-0 runs.
cd "$(dirname "$0")/.." && mkdir -p logs
common="--H 16 --W 16 --iters 1000 --eval_every 50 --samplers 8 --queue 2 --turn_delay 0.05"
run() { CUDA_VISIBLE_DEVICES=$1 nohup python3 async_runner.py --device cuda --name "$2" "${@:3}" > "logs/$2.log" 2>&1 & }
while pgrep -u "$(whoami)" -f "name a_(coro_budget|tree_budget)_s0" > /dev/null; do sleep 60; done
for s in 1 2; do
  run 1 a_coro_budget_s$s --objective coro $common --seed $s --drift_budget 0.002 --beta_lr 1
  run 2 a_tree_budget_s$s --objective tree --groups 26 --group_size 4 --branch 4 $common --seed $s --drift_budget 0.002 --beta_lr 1
  run 1 a_ppo_s$s --objective ppo $common --seed $s
  wait
done
run 2 a_ppo_s0 --objective ppo $common --seed 0
wait
