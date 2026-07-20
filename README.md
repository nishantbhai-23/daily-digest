# daily-digest — Chief of Staff AI

Turns a day's worth of email, calendar events, notes, and tasks into one
daily brief: what actually needs your attention today, what you're at risk
of dropping, and a handful of ready-to-send drafts for anything answerable
in under a minute. Multi-tenant, provider-agnostic (Ollama, Anthropic,
Google, OpenRouter, DeepSeek), stdlib-only — no `pip install` required.

For the architecture and the reasoning behind it, see
[`docs/HIGH_LEVEL_DESIGN.md`](docs/HIGH_LEVEL_DESIGN.md). This doc is the
practical "how do I run it" reference.

## Requirements

- Python 3.10+
- An API key for at least one provider (Anthropic, Google, OpenRouter, or
  DeepSeek), **or** a local [Ollama](https://ollama.com) install — no key
  needed, but slower and less reliable at following JSON-schema instructions.
- Nothing to `pip install`. Every script is plain-stdlib Python.

## 1. Set up API keys

```bash
cp .env.example .env
# edit .env, fill in the key(s) you have
```

`.env` is gitignored and loaded automatically (`digest/core/env.py`, wired
into `digest/core/llm.py` at import time) — no shell export needed, and a
real environment variable always overrides the file if you'd rather set
one that way.

## 2. Quickstart — run the pipeline once

The repo ships with a synthetic dataset already generated
(`data/inbox/`, `data/calendar/calendar.ics`, `data/notes/`,
`data/tasks/tasks.json`, `data/persona.md`) so you can run the full
pipeline immediately:

```bash
python3 -m digest.agents.triage_agent   --provider deepseek --model deepseek-chat
python3 -m digest.agents.calendar_agent --provider deepseek --model deepseek-chat
python3 -m digest.agents.notes_agent    --provider deepseek --model deepseek-chat
python3 -m digest.orchestrator          --provider deepseek --model deepseek-chat
```

The first three build/update a rolling ledger per source (email, calendar,
notes — each incremental, safe to re-run). The orchestrator reads all
three ledgers plus `data/tasks/tasks.json` and writes the final brief to
`output/daily_brief.md` (also printed to stdout, and copied into
`output/history/` with a timestamp every run).

To regenerate the synthetic dataset from scratch (or with a different
random seed):

```bash
python3 generate_data.py --seed 42 --output-dir data
```

To use **your own real data** instead of the synthetic set, just populate
the same paths directly: `.eml` files in `data/inbox/`, one `.ics`
calendar at `data/calendar/calendar.ics`, `.md` notes in `data/notes/`,
and `data/tasks/tasks.json`. Write your own `data/persona.md` describing
your priorities, tone, and what a good digest means to you — see
`data/tenants/arclight/persona.md` or any `generate_persona.py` output for
the expected structure.

## 3. The three MAP-REDUCE agents

Each of `triage_agent.py` / `calendar_agent.py` / `notes_agent.py` shares
the same flag set:

| Flag | Default | Meaning |
|---|---|---|
| `--provider` | `ollama` | `ollama` \| `anthropic` \| `google` \| `openrouter` \| `deepseek` |
| `--model` | `llama3` | Model name for the chosen provider |
| `--temperature` | `0.0` | LLM sampling temperature |
| `--map-only` | off | Run only extraction (skip weekly-summary compaction) |
| `--reduce-only` | off | Only re-synthesize from the existing ledger, no new extraction |
| `--workers` | `4` | Concurrent MAP-phase workers |
| `--tenant` | `default` | See [Multi-tenancy](#4-multi-tenancy) below |

`triage_agent.py` and `calendar_agent.py` additionally support incremental/
cold-hot ingestion (see [Cold + hot ingestion](#6-cold--hot-ingestion)
below); `notes_agent.py` doesn't need it — it always processes whatever
`.md` files are present, deduped by filename via the ledger, so re-running
it after adding one new note file only processes that new file.

## 4. Multi-tenancy

Every script accepts `--tenant <id>`. `--tenant default` (or omitting the
flag) uses today's flat layout (`data/...`, `output/...`); any other id
resolves under `data/tenants/<id>/` and `output/tenants/<id>/` — a
completely isolated sandbox, same file layout either way:

```
data/tenants/<id>/
  persona.md
  tenant_config.json
  inbox/            *.eml
  calendar/calendar.ics
  notes/            *.md
  tasks/tasks.json

output/tenants/<id>/
  daily_brief.md
  history/          timestamped past briefs
  rolling_ledger.json, calendar_rolling_ledger.json, notes_rolling_ledger.json
  metrics.jsonl
```

Run the same 4 commands from the quickstart with `--tenant <id>` appended
to target a specific tenant.

### `tenant_config.json` — behavioral toggles

```json
{
  "use_persona_in_map": true,
  "never_draft_contacts": [{"name": "Sam Park", "email": "sam@example.com"}],
  "map_noise_filter": {
    "blocked_senders": ["spam@vendor.com"],
    "blocked_domains": ["newsletter-fake.com"]
  }
}
```
- `use_persona_in_map` — whether the persona is injected into MAP-phase
  (per-day extraction) prompts too, not just REDUCE/synthesis (which
  always get it). Off means faster/cheaper MAP calls with no priority
  judgment at that stage.
- `never_draft_contacts` — matched by name and email against every
  dispatchable item before Stage 3 drafts anything; a match is surfaced
  for you to handle personally, never auto-drafted. Enforced in code, not
  left to prompt compliance.
- `map_noise_filter` — senders/domains filtered out before they ever reach
  an LLM call (email only).

### `data/system_config.json` — fleet-level ceilings

One file, shared across every tenant, not tenant-editable (mirrors
`tenant_config.json`'s shape but lives one level up):
```json
{
  "max_qps_per_tenant": 2.0,
  "allowed_providers": ["ollama", "anthropic", "google", "openrouter", "deepseek"],
  "circuit_breaker_threshold": 5
}
```

### Running a fleet of tenants concurrently

```bash
python3 -m digest.run_fleet --tenants acme globex initech --provider deepseek --model deepseek-chat
```
`--seed-from-default` copies the `default` tenant's data + ledgers into
each named tenant first, as a convenience for emulating a fleet without
authoring N separate datasets by hand.

## 5. The orchestrator

```bash
python3 -m digest.orchestrator --provider deepseek --model deepseek-chat --tenant default --synthesis-variant single_call
```

| Flag | Default | Meaning |
|---|---|---|
| `--provider` / `--model` / `--temperature` | `ollama` / `llama3` / `0.0` | Same as the agents |
| `--tenant` | `default` | Must match the `--tenant` the three agents were run with |
| `--synthesis-variant` | `single_call` | `single_call` (one LLM call for the whole brief) or `staged` (narrative written first, dispatchable items as a dependent second call — see `digest/eval/eval_synthesis_variants.py` for the comparison this default is pending on) |

Stages: deterministic cross-reference index → contradiction detection →
unified synthesis → draft generation → assembly (pure Python). See
`docs/HIGH_LEVEL_DESIGN.md`'s "Core architectural decisions" for why it's
staged this way rather than one big prompt.

## 6. Cold + hot ingestion

`triage_agent.py` and `calendar_agent.py` support three extra flags for
simulating/bounding incremental arrival, all operating on whatever days
are present in the source data (no new files needed to test with):

| Flag | Meaning |
|---|---|
| `--digest-days N` | Cap a run to the most recent N days found in the source |
| `--holdout-days N` | Exclude the N most recent days from this run — a later run without the flag picks them up as "new" through the normal incremental ledger diff |
| `--hot-input PATH` | Point at an extra directory (`.eml`) or file (`.ics`) of newly-arrived data, tagged `hot` in the ledger and flagged 🔥 in output |

## 7. Citations — "where did this line come from"

A separate, optional post-processing pass — doesn't touch any prompt, run
it any time after the orchestrator has produced a brief:

```bash
python3 -m digest.core.citations --tenant default --provider deepseek --model deepseek-chat
python3 -m digest.core.citations --tenant default --keyword-only   # free, no LLM call, lower recall
```

Reads `output/tenants/<id>/daily_brief.md`, matches each bullet against
the day's real source emails/events/notes (keyword matching first, one
batched LLM-judge call for anything keyword matching can't confidently
place, every claimed match verified against the real candidate list before
being trusted), and writes an annotated copy to
`output/tenants/<id>/daily_brief_cited.md` — e.g.
`...you need to: _[source: email: 0002.eml, notes: 2026-07-18-field-note.md]_`.
A bullet with no confident match is left uncited rather than guessing.

## 8. Small-scale human review demo

For sanity-checking the pipeline on a dataset small enough to read
end-to-end by hand (5-10 emails, 2-3 calendar events with a genuine
conflict, 2-3 notes, 2 tasks, revealed incrementally across 3 simulated
days) rather than trusting a 30-day batch on faith:

```bash
python3 generate_persona.py --tenant-id demo-1 --provider deepseek --model deepseek-chat
# optional: --hint "boutique law firm, managing partner" to steer flavor

./scripts/run_review_step.sh demo-1 1   # reveals day 1
./scripts/run_review_step.sh demo-1 2   # reveals days 1-2
./scripts/run_review_step.sh demo-1 3   # reveals all 3 days
```

Each step prints where the brief landed. `generate_persona.py` also writes
`data/tenants/demo-1/ANSWER_KEY.md` — what you should expect to see at
each step, so you can check the real output against it rather than
guessing whether it's right. Full walkthrough:
[`docs/QUICKSTART_REVIEW.md`](docs/QUICKSTART_REVIEW.md). Run
`generate_persona.py` again with a different `--tenant-id` any time you
want a fresh fictional company/persona to test against.

## 9. Evaluation tooling

None of these run automatically — each calls a real model and costs real
provider budget.

```bash
# MAP-phase extraction accuracy against known planted scenarios
python3 -m digest.eval.eval_map --provider deepseek --model deepseek-chat [--source email|calendar|notes|all]

# Zero-shot vs. few-shot MAP prompt comparison
python3 -m digest.eval.eval_prompt_variants --provider deepseek --model deepseek-chat --trials 5

# single_call vs. staged Stage-2 synthesis comparison
python3 -m digest.eval.eval_synthesis_variants --tenant default --provider deepseek --model deepseek-chat --trials 5
```

Every run of the last two is recorded to `eval_history/results.jsonl` (with
a content-addressed snapshot of the exact prompt text tested under
`eval_history/prompts/`), so "did accuracy hold up after switching models"
is a query over history, not a lost one-off result.

## 10. Tests

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```
Fully offline — no LLM calls, no API keys needed. Everything LLM-dependent
lives in the eval scripts above instead, run manually.

## Project layout

```
digest/
  parsers/     email_parser.py, calendar_parser.py, notes_parser.py, tasks_parser.py
  agents/      triage_agent.py, calendar_agent.py, notes_agent.py  (MAP-REDUCE per source)
  core/        ledger.py, llm.py, persona.py, tenant_config.py, tenant_paths.py,
               resilience.py, cross_reference.py, citations.py, tasks_signals.py, env.py
  eval/        golden_scenarios.py, eval_map.py, eval_prompt_variants.py,
               eval_synthesis_variants.py, eval_history.py, digest_checks.py
  orchestrator.py
  run_fleet.py

data/                    "default" tenant's input data
data/tenants/<id>/       other tenants' input data
output/, output/tenants/<id>/    generated ledgers, briefs, history
eval_history/            persistent eval results + prompt snapshots
scripts/run_review_step.sh
generate_data.py, generate_persona.py
docs/                    HIGH_LEVEL_DESIGN.md, LOW_LEVEL_DESIGN.md, ERROR_HANDLING.md,
                          SECURITY.md, QUICKSTART_REVIEW.md
```

## Further reading

- [`docs/HIGH_LEVEL_DESIGN.md`](docs/HIGH_LEVEL_DESIGN.md) — architecture, every major decision and why, cost/latency, reliability philosophy.
- [`docs/LOW_LEVEL_DESIGN.md`](docs/LOW_LEVEL_DESIGN.md) — implementation-level detail, planted eval scenarios.
- [`docs/ERROR_HANDLING.md`](docs/ERROR_HANDLING.md) — retry/circuit-breaker/grounding-check design.
- [`docs/SECURITY.md`](docs/SECURITY.md) — tenant isolation, path-traversal guards, what's out of scope.
- [`docs/QUICKSTART_REVIEW.md`](docs/QUICKSTART_REVIEW.md) — the human-review demo, in full.
