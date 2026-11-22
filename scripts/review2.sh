#!/bin/bash
# Fourth review, the reader's side (2026-11-22). Settles what three seeds evaluated every 50 updates could not:
#  T256  Branched CORO with 256 rewarded episodes per update (64 trunks x 4), the same count as E4. If the
#        sibling contrast adds signal per episode, T256 reaches 0.9 well before E4.
#  n_*   E4, E5, C1 (same forked data, flat CORO), C2 (416 independent episodes) and T256 on 8 seeds,
#        evaluated every 10 updates. Evaluation has its own random stream, so seeds 0-2 retrain bit for bit.
#  p_*   PPO-clip tuned as CORO was: advantages standardised (A0) or only centred like CORO's targets (A0dr),
#        clip 0.1/0.2/0.3, chosen on seed 100 only; flat and branched (B1).
#  x_e7glob  beta held at 0.3, drift enforced by one separate KL multiplier (the global form of X1).
# At most 12 runs at a time, GPUs 1-3 in turn.
cd "$(dirname "$0")/.." && mkdir -p logs
common="--H 16 --W 16 --iters 1000 --U 4 --qmax 8"
budget="--drift_budget 0.002 --beta_lr 1"
tree="--groups 26 --group_size 4 --branch 4"
jobs() {
  for s in 0 1 2 3 4 5 6 7; do echo "n_T256_s$s --objective tree --groups 16 --group_size 4 --branch 4 $common $budget --eval_every 10 --seed $s"; done
  for s in 0 1 2 3 4 5 6 7; do
    echo "n_E4_s$s --objective coro $common $budget --eval_every 10 --seed $s"
    echo "n_E5_s$s --objective tree $tree $common $budget --eval_every 10 --seed $s"
  done
  for c in A0 A0dr; do for clip in 0.1 0.2 0.3; do
    echo "p_ppo_${c}_clip${clip}_s100 --objective ppo --credit $c --clip $clip $common --eval_every 50 --seed 100"
    echo "p_treeppo_${c}_clip${clip}_s100 --objective tree_ppo $tree --credit $c --clip $clip $common --eval_every 50 --seed 100"
  done; done
  for lr in 100 1000 10000; do for s in 100 0 1 2; do
    echo "x_e7glob_lr${lr}_s$s --objective tree $tree $common --eval_every 50 --drift_budget 0.002 --beta_lr 0 --turn_penalty_lr $lr --turn_penalty_global --seed $s"
  done; done
  for s in 0 1 2 3 4 5 6 7; do
    echo "n_C1_s$s --objective tree_flat $tree $common $budget --eval_every 10 --seed $s"
    echo "n_C2_s$s --objective coro --groups 26 --group_size 16 $common $budget --eval_every 10 --seed $s"
  done
}
jobs | awk '{print (NR - 1) % 3 + 1, $0}' |
  xargs -P 12 -L 1 sh -c 'CUDA_VISIBLE_DEVICES=$0 python3 runner.py --device cuda --name "$@" > "logs/$1.log" 2>&1'
