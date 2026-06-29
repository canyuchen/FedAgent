#!/bin/bash
# Generic: wait for a fed run to finish. args: <name> <outdir>
name="$1"; d="$2"
for i in $(seq 1 120); do
  if [ -f "$d/federated_summary.json" ]; then echo "${name}_DONE summary present after ~$((i*30))s"; exit 0; fi
  if ! pgrep -f "run_fed.*${name}" >/dev/null 2>&1; then
    sleep 5
    if [ -f "$d/federated_summary.json" ]; then echo "${name}_DONE (proc exited, summary present)"; exit 0; fi
    echo "${name}_PROC_GONE but NO summary -> possible crash; rounds:"; ls -1d $d/round_* 2>/dev/null | tail -3; exit 2
  fi
  sleep 30
done
echo "${name}_WATCH_TIMEOUT (~60min)"; exit 3
