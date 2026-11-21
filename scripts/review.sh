#!/bin/bash
# Runs added after the third review, seeds 0-2. Needs the r_E4/r_E5 reruns from controls.sh (for --beta_path).
#   B1 tree_ppo        same forked data as E5, PPO-clip on the advantages Branched CORO's two levels imply,
#                      no budget: is the speed-up from branching or from CORO? (compare f_tree_beta0.3, E1)
#   S1-S3 sync_*_mb4   synchronous, each batch split into 4 whole-group steps, so from the second step on the
#                      stored log-probs are behind the trainer: does sync lose to stale because l = 0?
#                      CORO and PPO-clip get an anchor / a clip that acts; plain policy gradient is the control
#   M1-M3 *_bp         the mechanism controls again, on the same random stream as their reference run and
#                      with its beta path replayed, so neither the data nor the dual can differ by accident
cd "$(dirname "$0")/.." && mkdir -p logs
GPUS=(1 2 3); i=0  # edit to match your machine
launch() { CUDA_VISIBLE_DEVICES=${GPUS[$((i % ${#GPUS[@]}))]} nohup python3 runner.py --device cuda --name "$1" "${@:2}" > "logs/$1.log" 2>&1 & i=$((i + 1)); }
common="--H 16 --W 16 --iters 1000 --eval_every 50"
stale="--U 4 --qmax 8"
budget="--drift_budget 0.002 --beta_lr 1"
shape="--groups 26 --group_size 4 --branch 4"
for s in 0 1 2; do
  launch v_tree_ppo_s$s --objective tree_ppo $shape $stale $common --seed $s
  for o in coro ppo none; do
    launch v_sync_${o}_mb4_s$s --objective $o --sync --minibatches 4 $common --seed $s
  done
  launch v_tree_lvl2_bp_s$s --objective tree $shape $stale $common $budget --tree_no_level1 --beta_path r_E5_tree_budget_s$s --seed $s
  launch v_tree_resample_bp_s$s --objective tree $shape $stale $common $budget --fork_resample --beta_path r_E5_tree_budget_s$s --seed $s
  launch v_coro_crn_bp_s$s --objective coro $stale $common $budget --crn --beta_path r_E4_coro_budget_s$s --seed $s
done
wait
