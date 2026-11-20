#!/bin/bash
# E7 = E5 (Branched CORO + scalar drift budget) + a separate KL penalty per staleness bucket, with its own
# multiplier lam[s] set by dual ascent. Fixes E6, where the per-bucket beta also scaled each turn's
# contrastive gradient and starved the fresh, rewarded final turn.
# The multiplier step size is chosen on seed 100 only; all three are also run on seeds 0-2 and all are
# reported, so the choice can be checked.
cd "$(dirname "$0")/.." && mkdir -p logs
GPUS=(1 2 3); i=0
launch() { CUDA_VISIBLE_DEVICES=${GPUS[$((i % 3))]} nohup python3 runner.py --device cuda --name "$1" "${@:2}" > "logs/$1.log" 2>&1 & i=$((i + 1)); }
args="--objective tree --groups 26 --group_size 4 --branch 4 --U 4 --qmax 8 --H 16 --W 16 --iters 1000 --eval_every 50 --drift_budget 0.002 --beta_lr 1"
for lr in 100 1000 10000; do
  for s in 100 0 1 2; do
    launch e7_lr${lr}_s$s $args --seed $s --turn_penalty_lr $lr
  done
done
wait
