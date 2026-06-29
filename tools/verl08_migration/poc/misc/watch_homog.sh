#!/bin/bash
# Wait for the homog run to finish (federated_summary.json appears OR run_fed pid exits),
# then report. Lightweight: sleeps, checks a file. ~50 min safety cap.
d=/tmp/xbb9020_fedagent_homog_c4
for i in $(seq 1 100); do
  if [ -f "$d/federated_summary.json" ]; then echo "HOMOG_DONE summary present after ~$((i*30))s"; ls -la $d/round_8/aggregated/hf 2>/dev/null; exit 0; fi
  if ! pgrep -f "run_fed.*homog" >/dev/null 2>&1; then
    sleep 5
    if [ -f "$d/federated_summary.json" ]; then echo "HOMOG_DONE (proc exited, summary present)"; exit 0; fi
    echo "HOMOG_PROC_GONE but NO summary -> possible crash; round dirs:"; ls -1d $d/round_* 2>/dev/null | tail -3; exit 2
  fi
  sleep 30
done
echo "WATCH_TIMEOUT (still running after ~50min)"; exit 3
