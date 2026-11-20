#!/bin/bash
# Day 2 sweep (tuning seed 100, H=16/W=16, sync): why does CORO sit on the plateau where A0 escapes?
cd "$(dirname "$0")/.." && mkdir -p logs
GPUS=(1 2 3); i=0
launch() { CUDA_VISIBLE_DEVICES=${GPUS[$((i % 3))]} nohup python3 runner.py --device cuda --name "$1" "${@:2}" > "logs/$1.log" 2>&1 & i=$((i + 1)); }
common="--H 16 --W 16 --iters 1500 --sync --seed 100 --eval_every 25"
for b in 0.03 0.1 0.3 1.0; do launch "sw_coro_b${b}_lr1e-3" --objective coro --beta $b $common; done
launch "sw_coro_b0.1_lr3e-3" --objective coro --beta 0.1 --lr 0.003 $common
launch "sw_none_A0_lr3e-3" --objective none --lr 0.003 $common
launch "sw_none_A0dr_lr1e-3" --objective none --credit A0dr $common
launch "sw_none_A0_lr1e-3" --objective none $common
wait
