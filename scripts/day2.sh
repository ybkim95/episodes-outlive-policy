#!/bin/bash
# Day 2 (env v2 + forced final reply, tuning seeds only). usage: day2.sh a|b
# machine a (GPUs 1-3): learnability, sync. machine b (GPUs 0-3): beta sensitivity + staleness preview.
cd "$(dirname "$0")/.." && mkdir -p logs
GPUS=($([ "$1" = a ] && echo 1 2 3 || echo 0 1 2 3)); i=0
PY=$([ "$1" = a ] && echo python3 || echo /tmp/takehome-venv/bin/python)  # machine b system torch is 1.13; venv on local disk
launch() { CUDA_VISIBLE_DEVICES=${GPUS[$((i % ${#GPUS[@]}))]} nohup $PY runner.py --device cuda --name "$1" "${@:2}" > "logs/$1.log" 2>&1 & i=$((i + 1)); }
if [ "$1" = a ]; then
  for s in 100 101; do
    for obj in none coro; do
      launch "v3_${obj}_H16_W16_sync_s$s" --objective $obj --H 16 --W 16 --iters 600 --sync --seed $s
      launch "v3_${obj}_H64_W8_sync_s$s" --objective $obj --H 64 --W 8 --iters 800 --sync --seed $s
    done
  done
else
  for b in 0.03 0.3; do launch "v3_coro_b${b}_H16_W16_sync_s100" --objective coro --beta $b --H 16 --W 16 --iters 600 --sync --seed 100; done
  for obj in none ppo oapl coro; do
    launch "v3_${obj}_H64_W8_U8q8_s100" --objective $obj --H 64 --W 8 --U 8 --qmax 8 --iters 800 --seed 100
  done
fi
wait
