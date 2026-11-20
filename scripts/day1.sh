#!/bin/bash
# Day 1 (tuning seeds only): A0 lr check at H=16/W=16, then the long-horizon check at H=64/W=8.
# Run on GPUs 1-3 of a shared machine (GPU 0 is used by others).
cd "$(dirname "$0")/.." && mkdir -p logs
i=0
launch() { CUDA_VISIBLE_DEVICES=$((1 + i % 3)) nohup python3 runner.py --device cuda --name "$1" "${@:2}" > "logs/$1.log" 2>&1 & i=$((i + 1)); }
for s in 100 101; do
  for lr in 0.0003 0.001; do launch "A0_H16_W16_lr${lr}_s$s" --H 16 --W 16 --lr $lr --seed $s; done
  launch "A0_H64_W8_lr0.001_s$s" --H 64 --W 8 --lr 0.001 --iters 600 --seed $s
done
wait
