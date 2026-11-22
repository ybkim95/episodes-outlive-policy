#!/bin/bash
# PPO-clip tuning, stage 2. Stage 1 (p_* in review2.sh, seed 100 only) chose centred-only advantages (A0dr) with
# clip 0.2 for both flat and branched PPO-clip: flat 501 updates to 0.9 against 651 for the default (standardised,
# clip 0.2); branched 301 against 401 (clip 0.2 and 0.3 tied at 301; the tie keeps the default clip). This runs
# the chosen settings on seeds 0-2, then the rest of the review queue (C1/C2 seeds the first launcher had not
# started). GPUs 1-2, 2 CPU threads per run (see resume.sh).
cd "$(dirname "$0")/.." && mkdir -p logs
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
common="--H 16 --W 16 --iters 1000 --U 4 --qmax 8 --eval_every 50"
tree="--groups 26 --group_size 4 --branch 4"
for s in 0 1 2; do
  echo "p2_ppo_A0dr_clip0.2_s$s --objective ppo --credit A0dr --clip 0.2 $common --seed $s"
  echo "p2_treeppo_A0dr_clip0.2_s$s --objective tree_ppo $tree --credit A0dr --clip 0.2 $common --seed $s"
done | awk '{print (NR - 1) % 2 + 1, $0}' |
  xargs -P 6 -L 1 sh -c 'CUDA_VISIBLE_DEVICES=$0 python3 runner.py --device cuda --name "$@" > "logs/$1.log" 2>&1'
bash scripts/resume.sh scripts/jobs_prio.txt 6
