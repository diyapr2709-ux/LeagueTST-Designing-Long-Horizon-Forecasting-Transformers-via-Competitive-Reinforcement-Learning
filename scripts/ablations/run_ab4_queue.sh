#!/usr/bin/env bash
# AB4 queue on the local RTX (docs/prereg_ablations.md amendment AB4): Anirudh's registered WP3 items, never launched
# on Modal (AB3), run here with the local references, one job at a time (--per-gpu 1), short items first and
# ABL_G_steps120_local last, each item and table logging to its own new file under results/logs/.
# Launch detached so it survives the session:  setsid nohup bash scripts/ablations/run_ab4_queue.sh >/dev/null 2>&1 &
# Progress: results/logs/ab4_queue_status.txt (one line per start and finish). A relaunch keeps finished work
# (runner.run_plan skips finished adapters and deployed seeds); an item marked DONE (exit 0 and an empty or
# absent errors.jsonl) is skipped; a PARTIAL or FAIL item runs again. It holds the same lock as the AB2 queue
# (scripts/ablations/run_l4_queue.sh), so only one job uses the card at a time, and it refuses a dirty tree, so
# every record's git hash is a commit.
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
LOGS=results/logs
STATUS=$LOGS/ab4_queue_status.txt
mkdir -p "$LOGS"

RUNS=results/runs/llm_local
exec 9> "$LOGS/l4_queue.lock"
flock -n 9 || { echo "$(date -Is) ABORT another queue holds $LOGS/l4_queue.lock" >> "$STATUS"; exit 1; }
if [ -n "$(git status --porcelain -- leaguetst scripts)" ]; then
  echo "$(date -Is) ABORT uncommitted changes under leaguetst/ or scripts/" >> "$STATUS"; exit 1
fi
shasum -a 256 -c docs/e4_rerun.sha256 > "$LOGS/ab4_queue_digests_$(date +%Y%m%dT%H%M%S).txt" 2>&1 || {
  echo "$(date -Is) ABORT e4_rerun.sha256 check failed" >> "$STATUS"; exit 1; }
echo "$(date -Is) QUEUE_START git $(git rev-parse HEAD)" >> "$STATUS"

run() {   # run <name> <output folder under $RUNS> <args...>
  local name=$1 out=$RUNS/$2; shift 2
  if grep -q " DONE $name " "$STATUS" 2>/dev/null; then return 0; fi
  local log=$LOGS/$name.log
  [ -e "$log" ] && log=$LOGS/${name}_relaunch$(date +%Y%m%dT%H%M%S).log
  echo "$(date -Is) START $name log=$log" >> "$STATUS"
  "$PY" scripts/ablate_llm.py "$@" --per-gpu 1 > "$log" 2>&1
  local rc=$?
  local errs=0
  [ -s "$out/errors.jsonl" ] && errs=$(grep -c . "$out/errors.jsonl")
  if [ $rc -ne 0 ]; then echo "$(date -Is) FAIL $name rc=$rc" >> "$STATUS"
  elif [ "$errs" -gt 0 ]; then echo "$(date -Is) PARTIAL $name errors=$errs ($out/errors.jsonl)" >> "$STATUS"
  else echo "$(date -Is) DONE $name rc=0" >> "$STATUS"; fi
}

Q15=Qwen/Qwen2.5-1.5B-Instruct
for d in ILI ETTh1; do
  t=$([ $d = ILI ] && echo ILI_T24 || echo ETTh1_T336)
  run "abl_p_anon_local_$d"   ABL_P_anon_local/$t   prompted --dataset $d --seeds 30 --model $Q15 --anonymous --item ABL_P_anon_local
  run "abl_p_repair_local_$d" ABL_P_repair_local/$t prompted --dataset $d --seeds 30 --model $Q15 --no-constrained --item ABL_P_repair_local
done
run abl_g_steps30_local  ABL_G_steps30_local/ILI_T24  transfer --source ETTh1 --target ILI --arms league:10 --steps 30 --item ABL_G_steps30_local
run abl_g_steps120_local ABL_G_steps120_local/ILI_T24 transfer --source ETTh1 --target ILI --arms league:10 --steps 120 --item ABL_G_steps120_local
echo "$(date -Is) QUEUE_END" >> "$STATUS"
