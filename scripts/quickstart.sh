#!/bin/bash
# One command that does the whole thing: generates a fresh fictional
# persona + dataset, runs the full pipeline against it (all 3 days
# visible), then attaches source citations to the result. Nothing to
# configure beyond an API key in .env (see README "1. Set up API keys") —
# everything else defaults to something that just works.
#
# Usage: ./scripts/quickstart.sh [tenant-id] [provider] [model]
#   ./scripts/quickstart.sh                              # random tenant id, deepseek/deepseek-chat
#   ./scripts/quickstart.sh my-test anthropic claude-3-5-sonnet-20241022

set -e

TENANT="${1:-quickstart-$(date +%s)}"
PROVIDER="${2:-deepseek}"
MODEL="${3:-deepseek-chat}"

echo "──────────────────────────────────────────────"
echo " Step 1/3 — generating a fresh persona + dataset"
echo "──────────────────────────────────────────────"
python3 generate_persona.py --tenant-id "$TENANT" --provider "$PROVIDER" --model "$MODEL"

echo ""
echo "──────────────────────────────────────────────"
echo " Step 2/3 — running the full pipeline"
echo "──────────────────────────────────────────────"
./scripts/run_review_step.sh "$TENANT" 3 "$PROVIDER" "$MODEL"

echo ""
echo "──────────────────────────────────────────────"
echo " Step 3/3 — attaching source citations"
echo "──────────────────────────────────────────────"
python3 -m digest.core.citations --tenant "$TENANT" --provider "$PROVIDER" --model "$MODEL"

echo ""
echo "Done. Everything for tenant '$TENANT':"
echo "  Raw data:    data/tenants/$TENANT/         (emails, calendar, notes, tasks, persona)"
echo "  Answer key:  data/tenants/$TENANT/ANSWER_KEY.md"
echo "  Brief:       output/tenants/$TENANT/daily_brief.md"
echo "  Cited brief: output/tenants/$TENANT/daily_brief_cited.md"
