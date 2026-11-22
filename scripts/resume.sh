#!/bin/bash
# On 2026-11-22 GPU 3 filled up with another user's job, and the runs that review2.sh and lowprec.sh had put
# on it failed at start (CUDA out of memory). Those queues were stopped (their running jobs kept going).
# This re-queues every job of the list in $1 (one "name flags..." per line, the jobs() of both scripts)
# that has not started or that failed, on GPUs 1-2 only, at most $2 (default 10) at a time.
# Later the same day, the rest of the queue was re-ordered by priority (jobs_prio.txt) and capped at 2 CPU threads
# per run: 20+ runs with torch's default of one thread per core left each run at ~6.5 s per update. The model
# runs on the GPU, so the thread count does not change results.
cd "$(dirname "$0")/.." && mkdir -p logs
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
while read -r name rest; do
  if [ ! -f "logs/$name.log" ] || grep -q -E "out of memory|Traceback" "logs/$name.log"; then echo "$name $rest"; fi
done < "$1" | awk '{print (NR - 1) % 2 + 1, $0}' |
  xargs -P "${2:-10}" -L 1 sh -c 'CUDA_VISIBLE_DEVICES=$0 python3 runner.py --device cuda --name "$@" > "logs/$1.log" 2>&1'
