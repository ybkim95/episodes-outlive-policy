#!/bin/bash
# The main comparison again, with a real asynchronous sampler and trainer (async_runner.py): 8 CPU sampler
# processes per run, a queue of 2 batches, weights swapped at turn boundaries, staleness measured per turn.
# Same env, batch sizes, beta, learning rate and drift budget as E4/E5 (all chosen on seed 100 in the
# simulated setting; nothing re-tuned). The three methods of one seed run at the same time, so they share
# the machine's load: wall-clock and staleness are comparable within a seed, not across seeds.
cd "$(dirname "$0")/.." && mkdir -p logs
common="--H 16 --W 16 --iters 1000 --eval_every 50 --samplers 8 --queue 2 --turn_delay 0.05"
run() { CUDA_VISIBLE_DEVICES=$1 nohup python3 async_runner.py --device cuda --name "$2" "${@:3}" > "logs/$2.log" 2>&1 & }
for s in 0 1 2; do
  run 1 a_coro_budget_s$s --objective coro $common --seed $s --drift_budget 0.002 --beta_lr 1
  run 2 a_tree_budget_s$s --objective tree --groups 26 --group_size 4 --branch 4 $common --seed $s --drift_budget 0.002 --beta_lr 1
  run 3 a_ppo_s$s --objective ppo $common --seed $s
  wait
done
