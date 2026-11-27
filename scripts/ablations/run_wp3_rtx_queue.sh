#!/usr/bin/env bash
# WP3 on the local RTX (amendment AB4, docs/LEG3_soumitra.md): the AB-P anon and repair arms and the AB-G step arms
# that were registered for Modal and never launched there (AB3). Same card, launcher and rules as the AB2 queue
# (scripts/ablations/run_l4_queue.sh, unchanged): one job at a time, each item and table logging to its own new
# file under results/logs/, a relaunch keeps finished work, DONE items are skipped, PARTIAL and FAIL items rerun.
# Launch detached:  setsid nohup bash scripts/ablations/run_wp3_rtx_queue.sh >/dev/null 2>&1 &
# Progress: results/logs/wp3_rtx_queue_status.txt. No job runs after 28 Nov 18:00 ET (AB4). After every item the queue copies its record files to
# results/ablations/rtx/<item>/<table>/ (never to results/search/, which scripts/picks.py and e5_referee.py read).
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
LOGS=results/logs
STATUS=$LOGS/wp3_rtx_queue_status.txt
RUNS=results/runs/llm_local
DEST=results/ablations/rtx
mkdir -p "$LOGS"

# The AB2 queue's lock, so the two queues never share the card.
exec 9> "$LOGS/l4_queue.lock"
flock -n 9 || { echo "$(date -Is) ABORT another queue holds $LOGS/l4_queue.lock" >> "$STATUS"; exit 1; }
if [ -n "$(git status --porcelain -- leaguetst scripts)" ]; then
  echo "$(date -Is) ABORT uncommitted changes under leaguetst/ or scripts/" >> "$STATUS"; exit 1
fi
shasum -a 256 -c docs/e4_rerun.sha256 > "$LOGS/wp3_rtx_queue_digests_$(date +%Y%m%dT%H%M%S).txt" 2>&1 || {
  echo "$(date -Is) ABORT e4_rerun.sha256 check failed" >> "$STATUS"; exit 1; }
grep -q '^## Amendment AB4' docs/prereg_ablations.md || {
  echo "$(date -Is) ABORT HEAD does not contain amendment AB4" >> "$STATUS"; exit 1; }
# The run path is the AB2 code: no change under leaguetst/, in the launcher, llm_local.py or the pins since AB2.
git diff --quiet c2dda8dfa49c6f090816f516285792a00585a79d HEAD -- leaguetst scripts/ablate_llm.py scripts/llm_local.py \
  requirements-llm.txt || { echo "$(date -Is) ABORT run-path code differs from the AB2 commit" >> "$STATUS"; exit 1; }
# AB4 deadline: no job runs after 28 Nov 18:00 ET (GNU date, as on the WSL2 card).
DEADLINE=$(date -d 2026-11-28T23:00:00Z +%s)
echo "$(date -Is) QUEUE_START git $(git rev-parse HEAD) card $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1 | head -1)" >> "$STATUS"

bring_back() {   # bring_back <item> <table>: record files only; adapters stay under $RUNS (gitignored)
  local src=$RUNS/$1/$2 dst=$DEST/$1/$2 f
  mkdir -p "$dst"
  for f in "$src"/*_runs.jsonl "$src"/training.jsonl "$src"/errors.jsonl; do
    [ -e "$f" ] && cp -p "$f" "$dst/"
  done
  echo "$(date -Is) COPIED $1/$2 to $dst" >> "$STATUS"
}

run() {   # run <name> <item> <table> <args...>
  local name=$1 item=$2 table=$3; shift 3
  local out=$RUNS/$item/$table
  if grep -q " DONE $name " "$STATUS" 2>/dev/null; then return 0; fi
  local log=$LOGS/$name.log
  [ -e "$log" ] && log=$LOGS/${name}_relaunch$(date +%Y%m%dT%H%M%S).log
  local left=$((DEADLINE - $(date +%s)))
  if [ "$left" -le 0 ]; then echo "$(date -Is) DEADLINE $name not started" >> "$STATUS"; return 0; fi
  echo "$(date -Is) START $name log=$log" >> "$STATUS"
  timeout --kill-after=60 "$left" "$PY" scripts/ablate_llm.py "$@" --item "$item" --per-gpu 1 > "$log" 2>&1
  local rc=$?
  if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then echo "$(date -Is) DEADLINE $name stopped at 28 Nov 18:00 ET" >> "$STATUS"; fi
  local errs=0
  [ -s "$out/errors.jsonl" ] && errs=$(grep -c . "$out/errors.jsonl")
  if [ $rc -ne 0 ]; then echo "$(date -Is) FAIL $name rc=$rc" >> "$STATUS"
  elif [ "$errs" -gt 0 ]; then echo "$(date -Is) PARTIAL $name errors=$errs ($out/errors.jsonl)" >> "$STATUS"
  else echo "$(date -Is) DONE $name rc=0" >> "$STATUS"; fi
  bring_back "$item" "$table"
}

Q15=Qwen/Qwen2.5-1.5B-Instruct
# Tested ILI arms first (the two prompted items, then the two step arms), then the descriptive ETTh1 tables.
run abl_p_anon_local_ILI     ABL_P_anon_local     ILI_T24    prompted --dataset ILI --seeds 30 --model $Q15 --anonymous
run abl_p_repair_local_ILI   ABL_P_repair_local   ILI_T24    prompted --dataset ILI --seeds 30 --model $Q15 --no-constrained
run abl_g_steps30_local      ABL_G_steps30_local  ILI_T24    transfer --source ETTh1 --target ILI --arms league:10 --steps 30
run abl_g_steps120_local     ABL_G_steps120_local ILI_T24    transfer --source ETTh1 --target ILI --arms league:10 --steps 120
run abl_p_anon_local_ETTh1   ABL_P_anon_local     ETTh1_T336 prompted --dataset ETTh1 --seeds 30 --model $Q15 --anonymous
run abl_p_repair_local_ETTh1 ABL_P_repair_local   ETTh1_T336 prompted --dataset ETTh1 --seeds 30 --model $Q15 --no-constrained
echo "$(date -Is) QUEUE_END" >> "$STATUS"
