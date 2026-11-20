#!/bin/bash
# Staleness severity sweep, H=16 / W=16, seeds 0-2. The main matrix (U=4, q<=8, staleness 0..11) did not
# break token-level PPO-clip; here every turn comes from a newer sampler (U=1: one episode spans 16
# versions) plus a heavy-tailed queue delay q <= 16 or 48, so staleness reaches 31 or 63.
cd "$(dirname "$0")/.." && mkdir -p logs
GPUS=(1 2 3); i=0
launch() { CUDA_VISIBLE_DEVICES=${GPUS[$((i % 3))]} nohup python3 runner.py --device cuda --name "$1" "${@:2}" > "logs/$1.log" 2>&1 & i=$((i + 1)); }
common="--H 16 --W 16 --iters 1000 --eval_every 50 --U 1"
for q in 16 48; do for s in 0 1 2; do
  for obj in ppo oapl coro none; do launch sev_q${q}_${obj}_s$s --objective $obj --qmax $q $common --seed $s; done
done; done
wait
