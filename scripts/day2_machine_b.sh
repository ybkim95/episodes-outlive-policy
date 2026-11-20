#!/bin/bash
# Day 2 on machine b (tuning seed 100, sync): does Branched CORO learn, and is H=32/W=8 learnable?
# tree uses 26 groups x 4 trunks x 4 continuations ~ the agent-turn budget of 32 x 8 plain episodes.
cd "$(dirname "$0")/.." && mkdir -p logs
PY=/tmp/takehome-venv/bin/python  # machine b system torch is 1.13; venv on local disk
GPUS=(0 1 2 3); i=0
launch() { CUDA_VISIBLE_DEVICES=${GPUS[$((i % 4))]} nohup $PY runner.py --device cuda --name "$1" "${@:2}" > "logs/$1.log" 2>&1 & i=$((i + 1)); }
tree="--objective tree --groups 26 --group_size 4 --branch 4"
launch v3_tree_H16_W16_sync_s100 $tree --H 16 --W 16 --iters 1500 --sync --seed 100 --eval_every 25
launch v3_tree_H32_W8_sync_s100 $tree --H 32 --W 8 --iters 1000 --sync --seed 100 --eval_every 25
launch v3_tree_H64_W8_sync_s100 $tree --H 64 --W 8 --iters 800 --sync --seed 100
for obj in none coro; do launch v3_${obj}_H32_W8_sync_s100 --objective $obj --H 32 --W 8 --iters 1000 --sync --seed 100 --eval_every 25; done
wait
