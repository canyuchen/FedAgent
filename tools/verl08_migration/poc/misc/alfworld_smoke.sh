#!/bin/bash
# Standalone CPU smoke of the ported ALFWorld service (no GPU). Start -> /health -> /create -> /reset -> /step.
set +e
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent
export ALFWORLD_DATA="$HOME/.cache/alfworld"
export ALFWORLD_POOL_SIZE=1
export ALFWORLD_PORT=8131
echo "[smoke] node=$(hostname) launching service (pool=1, port=8131) ..."
bash "$REPO/fedagent/alfworld_service/run_service.sh" > /tmp/alfworld_smoke_service.log 2>&1 &
SVC=$!
echo "[smoke] service pid=$SVC; waiting for /health (up to 420s) ..."
ok=0
for i in $(seq 1 140); do
  if curl -sf http://localhost:8131/health >/tmp/alf_health.json 2>/dev/null; then ok=1; break; fi
  if ! kill -0 $SVC 2>/dev/null; then echo "[smoke] SERVICE DIED during warmup"; break; fi
  sleep 3
done
if [ "$ok" != "1" ]; then
  echo "[smoke] HEALTH FAILED. Last 50 lines of service log:"; tail -50 /tmp/alfworld_smoke_service.log
  kill $SVC 2>/dev/null; exit 1
fi
echo "[smoke] HEALTH OK:"; cat /tmp/alf_health.json; echo
python3 - <<'PY'
import json, urllib.request
B="http://localhost:8131"
def post(p, d):
    r=urllib.request.Request(B+p, data=json.dumps(d).encode(), headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=120).read())
sid="smoke1"
print("[create]", post("/create", {"session_id":sid}))
rs=post("/reset", {"session_id":sid, "seed":0})
obs=rs.get("obs",""); adm=rs.get("admissible_commands",[])
print("[reset] obs[:200]=", repr(obs[:200]))
print("[reset] #admissible=", len(adm), "first5=", adm[:5])
act=adm[0] if adm else "look"
st=post("/step", {"session_id":sid, "text":f"<think>smoke</think><action>{act}</action>"})
print(f"[step '{act}'] reward=", st.get("reward"), "done=", st.get("done"),
      "valid=", st.get("is_action_valid"), "success=", st.get("success"))
print("[step] obs[:200]=", repr((st.get("obs") or "")[:200]))
print("[step] #admissible_next=", len(st.get("admissible_commands",[])))
print("ALFWORLD SERVICE SMOKE: PASS")
PY
RC=$?
echo "[smoke] killing service pid=$SVC"; kill $SVC 2>/dev/null
exit $RC
