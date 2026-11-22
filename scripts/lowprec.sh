#!/bin/bash
# The regime CORO was built for: the sampler disagrees with the trainer even on fresh turns. The sampler runs
# 4-bit weights (blocks of 16, absmax scale; --sampler_bits 4), the trainer full precision, as with
# low-precision rollouts. At initialisation this alone raises per-token KL from ~1e-3 to ~8e-3 and halves
# the median trajectory ESS. No budget anywhere (beta fixed at 0.3), so the drift dual cannot confound it.
# References without rounding: E1, E1b, E3, the fixed-beta Branched CORO run, and sev_q16_*.
cd "$(dirname "$0")/.." && mkdir -p logs
common="--H 16 --W 16 --iters 1000 --eval_every 50 --sampler_bits 4"
jobs() {
  for s in 0 1 2; do
    for o in ppo coro none; do
      echo "q4_${o}_s$s --objective $o --U 4 --qmax 8 $common --seed $s"
      echo "q4sev_${o}_s$s --objective $o --U 1 --qmax 16 $common --seed $s"
    done
    echo "q4_tree_s$s --objective tree --groups 26 --group_size 4 --branch 4 --U 4 --qmax 8 $common --seed $s"
    echo "q4sev_tree_s$s --objective tree --groups 26 --group_size 4 --branch 4 --U 1 --qmax 16 $common --seed $s"
  done
}
jobs | awk '{print (NR - 1) % 3 + 1, $0}' |
  xargs -P 6 -L 1 sh -c 'CUDA_VISIBLE_DEVICES=$0 python3 runner.py --device cuda --name "$@" > "logs/$1.log" 2>&1'
