#!/bin/bash
# Day 2, full context (W = H), GPUs 1-3 of machine a (machine b became unreachable mid-run).
# W < H failed for every method: WRITE + recall is a two-part skill with no partial credit, so the
# policy collapses to always-PASS before exploring it. That is an exploration problem, out of scope here.
cd "$(dirname "$0")/.." && mkdir -p logs
GPUS=(1 2 3); i=0
launch() { CUDA_VISIBLE_DEVICES=${GPUS[$((i % 3))]} nohup python3 runner.py --device cuda --name "$1" "${@:2}" > "logs/$1.log" 2>&1 & i=$((i + 1)); }
tree="--objective tree --groups 26 --group_size 4 --branch 4"
for H in 32 64; do
  IT=$([ $H = 32 ] && echo 1000 || echo 800)
  launch v4_none_H${H}_W${H}_sync_s100 --objective none --H $H --W $H --iters $IT --sync --seed 100 --eval_every 25
  launch v4_coro_H${H}_W${H}_sync_s100 --objective coro --H $H --W $H --iters $IT --sync --seed 100 --eval_every 25
  launch v4_tree_H${H}_W${H}_sync_s100 $tree --H $H --W $H --iters $IT --sync --seed 100 --eval_every 25
done
wait
