#!/bin/bash
# Runs one step of the 3-step incremental review walkthrough for a small,
# hand-reviewable tenant (see generate_persona.py and docs/QUICKSTART_REVIEW.md).
#
# Only EMAIL is revealed incrementally (--holdout-days, pure slicing over
# whatever days are present — ledger.apply_digest_window, no new pipeline
# logic): step 1 reveals day 1 only, step 2 reveals days 1-2, step 3 reveals
# all 3 days. Calendar, notes, and tasks are always fully visible from
# Step 1 — deliberate, not a limitation: a real calendar shows next week's
# meetings today, and notes_agent.py has no day-windowing flag at all (it
# always processes every note file present, deduped by note_id via the
# ledger). Windowing calendar the same way as email also doesn't compose
# well with a small 2-3-event dataset — apply_digest_window only holds
# anything out when holdout_days is strictly less than the number of
# distinct days that actually have events, and 2-3 sparse events can't
# spread across 3 distinct days while still containing a same-day conflict
# pair, so it would silently degenerate into "hold out nothing" anyway.
#
# --map-only throughout: orchestrator.py reads the ledger directly, not the
# REDUCE-phase markdown summary, so REDUCE/compaction isn't needed at this
# scale.
#
# Usage: ./scripts/run_review_step.sh <tenant-id> <1|2|3> [provider] [model]

set -e

TENANT="$1"
STEP="$2"
PROVIDER="${3:-deepseek}"
MODEL="${4:-deepseek-chat}"

if [ -z "$TENANT" ] || [ -z "$STEP" ]; then
  echo "Usage: $0 <tenant-id> <1|2|3> [provider] [model]"
  exit 1
fi

case "$STEP" in
  1) HOLDOUT="--holdout-days 2" ;;
  2) HOLDOUT="--holdout-days 1" ;;
  3) HOLDOUT="" ;;
  *) echo "Usage: $0 <tenant-id> <1|2|3> [provider] [model]"; exit 1 ;;
esac

python3 -m digest.agents.triage_agent --tenant "$TENANT" --map-only $HOLDOUT --provider "$PROVIDER" --model "$MODEL"
python3 -m digest.agents.calendar_agent --tenant "$TENANT" --map-only --provider "$PROVIDER" --model "$MODEL"
python3 -m digest.agents.notes_agent --tenant "$TENANT" --map-only --provider "$PROVIDER" --model "$MODEL"
python3 -m digest.orchestrator --tenant "$TENANT" --provider "$PROVIDER" --model "$MODEL"

echo ""
echo "Step $STEP complete."
echo "Brief:      output/tenants/$TENANT/daily_brief.md"
echo "Answer key: data/tenants/$TENANT/ANSWER_KEY.md"
