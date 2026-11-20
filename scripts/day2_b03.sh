#!/bin/bash
# Day 2, beta=0.3 (picked on tuning seed 100, H=16 sweep). Learnability of CORO and Branched CORO, sync.
# tree uses 26 groups x 4 trunks x 4 continuations ~ the agent-turn budget of 32 x 8 plain episodes.
cd "$(dirname "$0")/.." && mkdir -p logs
PY=/tmp/takehome-venv/bin/python  # machine b system torch is 1.13; venv on local disk
GPUS=(0 1 2 3); i=1  # GPU 0 already runs v3_none_H32_W8_sync_s100
launch() { CUDA_VISIBLE_DEVICES=${GPUS[$((i % 4))]} nohup $PY runner.py --device cuda --beta 0.3 --name "$1" "${@:2}" > "logs/$1.log" 2>&1 & i=$((i + 1)); }
tree="--objective tree --groups 26 --group_size 4 --branch 4"
for s in 100 101; do launch v3_tree_b0.3_H16_W16_sync_s$s $tree --H 16 --W 16 --iters 1500 --sync --seed $s --eval_every 25; done
launch v3_coro_b0.3_H16_W16_sync_s101 --objective coro --H 16 --W 16 --iters 1500 --sync --seed 101 --eval_every 25
for H in 32 64; do
  launch v3_tree_b0.3_H${H}_W8_sync_s100 $tree --H $H --W 8 --iters 1000 --sync --seed 100 --eval_every 25
  launch v3_coro_b0.3_H${H}_W8_sync_s100 --objective coro --H $H --W 8 --iters 1000 --sync --seed 100 --eval_every 25
done
wait
