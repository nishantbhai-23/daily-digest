# daily-digest — Chief of Staff AI

Turns a day's worth of email, calendar events, notes, and tasks into one
daily brief: what actually needs your attention today, what you're at risk
of dropping, and a handful of ready-to-send drafts for anything answerable
in under a minute. Multi-tenant, provider-agnostic (Ollama, Anthropic,
Google, OpenRouter, DeepSeek) — standard-library only when using DeepSeek/OpenRouter, while others require SDK installs.

For the architecture and the reasoning behind it, see
[`docs/HIGH_LEVEL_DESIGN.md`](docs/HIGH_LEVEL_DESIGN.md). This doc is the
practical "how do I run it" reference.

## Requirements

- Python 3.10+
- An API key for at least one provider (Anthropic, Google, OpenRouter, or
  DeepSeek), **or** a local [Ollama](https://ollama.com) install — no key
  needed, but slower and less reliable at following JSON-schema instructions.
- Provider SDK dependencies (run `pip install -r requirements.txt` to install dependencies for the default Ollama provider, or install `anthropic` / `google-generativeai` if using them. DeepSeek and OpenRouter are fully standard-library-only with zero external dependencies).

## 1. Set up API keys

```bash
cp .env.example .env
# edit .env, fill in the key(s) you have
```

`.env` is gitignored and loaded automatically (`digest/core/env.py`, wired
into `digest/core/llm.py` at import time) — no shell export needed, and a
real environment variable always overrides the file if you'd rather set
one that way.

## 2. Quickstart — one command

The lowest-effort way to see the whole tool work, with nothing to set up
beyond your API key:

```bash
./scripts/quickstart.sh
```

Generates a fresh fictional persona + a small realistic dataset (emails,
calendar, notes, tasks), runs the full pipeline against it, and attaches
source citations — prints the finished brief to your terminal and tells
you where every file landed. Takes about 15-20 seconds and a few cents of
API spend on `deepseek-chat` (the default). Re-run it any time for a new
random persona/company, or pin one:

```bash
./scripts/quickstart.sh my-test-run anthropic claude-3-5-sonnet-20241022
#                        ^tenant-id  ^provider ^model
```

That's genuinely "it" for exploring — the rest of this doc explains how
each piece works and how to point the tool at your own real data.

## 3. Customizing for your use case

Everything below is one flag or one file away. Quick map of where to look:

| I want to... | See |
|---|---|
| Use my own real email/calendar/notes/tasks | [Bring your own data](#5-bring-your-own-data) below |
| Steer the generated persona's industry/role | `--hint "boutique law firm, managing partner"` on `generate_persona.py` |
| Change what gets suppressed as noise, or never auto-drafted | `tenant_config.json` — [Multi-tenancy](#6-multi-tenancy) |
| Run against a different LLM provider/model | `--provider` / `--model` on every script — [Requirements](#requirements) |
| Run several tenants concurrently (a fleet) | `digest.run_fleet` — [Multi-tenancy](#6-multi-tenancy) |
| Simulate data arriving incrementally, or a burst of new data | `--digest-days` / `--holdout-days` / `--hot-input` — [Cold + hot ingestion](#9-cold--hot-ingestion) |
| See which email/event/note a digest line actually came from | `digest.core.citations` — [Citations](#10-citations--where-did-this-line-come-from) |
| Compare synthesis approaches, or check extraction accuracy against known scenarios | [Evaluation tooling](#12-evaluation-tooling) |
| Sanity-check a small dataset by hand before trusting a real 30-day inbox | [Small-scale human review demo](#11-small-scale-human-review-demo) |
| Change *what counts as a good digest* at the prompt level | Persona is prose (`persona.md`) the model interprets; anything that must never depend on prompt compliance (never-draft contacts, blocked senders) belongs in `tenant_config.json` instead — see `digest/core/tenant_config.py`'s module docstring for why they're split |

## 4. Manual quickstart — step by step

The repo ships with a synthetic dataset already generated
(`data/inbox/`, `data/calendar/calendar.ics`, `data/notes/`,
`data/tasks/tasks.json`, `data/persona.md`) — this is what
`quickstart.sh` above runs under the hood, one command per stage, useful
once you want to understand or modify the pipeline instead of just
watching it run:

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

## 5. Bring your own data

Every source is a standard, widely-produced file format — nothing
proprietary to learn beyond `tasks.json`'s shape. Point a tenant's
directory (`data/tenants/<your-id>/`, see [Multi-tenancy](#6-multi-tenancy)
below — or the flat `data/` layout for the `default` tenant) at real files
in these shapes and the pipeline runs exactly the same as it does against
generated data:

| Path | Format | Notes |
|---|---|---|
| `inbox/*.eml` | Standard RFC 822 email (headers + body) | Parsed with Python's stdlib `email` module — whatever your mail client exports as `.eml` works as-is. Reads `Subject`, `From`, `To`, `Date`, `Message-ID`. |
| `calendar/calendar.ics` | Standard iCalendar, one or more `VEVENT`s | Export from Google Calendar / Outlook / Apple Calendar and drop it in unmodified. Reads `SUMMARY`, `DTSTART`/`DTEND`, `DESCRIPTION`, `LOCATION`, `ATTENDEE`, `STATUS`. |
| `notes/*.md` | Plain markdown, one file per note | Filename **must** follow `YYYY-MM-DD-slug.md` — the date drives freshness/windowing logic; anything else parses but falls back to an "unknown" date and won't be time-aware. A leading `# Title` line and `- [ ]`/`- [x]` checkboxes are picked up if present, but neither is required. |
| `tasks/tasks.json` | A JSON array of task objects | Required fields: `id`, `title`. Everything else is optional with a sane default if omitted — `status` (defaults `"todo"`), `priority` (defaults `"?"`), `due_date` (`YYYY-MM-DD`), `created_at`, `description`, `tags`, `blocked_by`, `subtasks` (list of `{"text", "done"}`). In practice you'll want `status`/`priority`/`due_date` set — they're what the digest actually reasons about. Unknown extra fields are ignored, not rejected. |
| `persona.md` | Freeform markdown, first person | **Required** — always injected at synthesis time regardless of any config. No enforced section structure, but `## People who matter` (name + priority per person) and `## What I don't want surfaced` are what the prompts lean on most; see `data/tenants/arclight/persona.md` or any `generate_persona.py` output for a full example. |
| `tenant_config.json` | JSON, see [Multi-tenancy](#6-multi-tenancy) | Optional — every key falls back to a sane default if the file is missing entirely. |

Once your files are in place, run the same commands as the manual
quickstart above with `--tenant <your-id>` appended (or the fleet runner
for several tenants at once) — no code changes needed. If you just want to
confirm your files parse before spending any API budget on them, the four
parser modules (`digest/parsers/*.py`) can be called directly and will
print a warning rather than crash on a malformed file:
```bash
python3 -c "from digest.parsers.email_parser import load_inbox; print(len(load_inbox('data/tenants/<your-id>/inbox')), 'emails parsed')"
```

## 6. Multi-tenancy

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

Run the same 4 commands from the [manual quickstart](#4-manual-quickstart--step-by-step)
with `--tenant <id>` appended to target a specific tenant, or
`./scripts/quickstart.sh <id>` to do the whole thing in one shot.

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

## 7. The three MAP-REDUCE agents

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
| `--tenant` | `default` | See [Multi-tenancy](#6-multi-tenancy) above |

`triage_agent.py` and `calendar_agent.py` additionally support incremental/
cold-hot ingestion (see [Cold + hot ingestion](#9-cold--hot-ingestion)
below); `notes_agent.py` doesn't need it — it always processes whatever
`.md` files are present, deduped by filename via the ledger, so re-running
it after adding one new note file only processes that new file.

## 8. The orchestrator

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

## 9. Cold + hot ingestion

`triage_agent.py` and `calendar_agent.py` support three extra flags for
simulating/bounding incremental arrival, all operating on whatever days
are present in the source data (no new files needed to test with):

| Flag | Meaning |
|---|---|
| `--digest-days N` | Cap a run to the most recent N days found in the source |
| `--holdout-days N` | Exclude the N most recent days from this run — a later run without the flag picks them up as "new" through the normal incremental ledger diff |
| `--hot-input PATH` | Point at an extra directory (`.eml`) or file (`.ics`) of newly-arrived data, tagged `hot` in the ledger and flagged 🔥 in output |

## 10. Citations — "where did this line come from"

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

## 11. Small-scale human review demo

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
want a fresh fictional company/persona to test against — or just use
`./scripts/quickstart.sh` (section 2) for the same idea skipped straight
to the fully-revealed end state.

## 12. Evaluation tooling

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

## 13. Tests

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
scripts/quickstart.sh, scripts/run_review_step.sh
generate_data.py, generate_persona.py
docs/                    HIGH_LEVEL_DESIGN.md, LOW_LEVEL_DESIGN.md, ERROR_HANDLING.md,
                          SECURITY.md, QUICKSTART_REVIEW.md
```

## Further reading

- [`docs/HIGH_LEVEL_DESIGN.md`](docs/HIGH_LEVEL_DESIGN.md) — the system as it stands today: architecture, pipeline stages, data model, reliability layer.
- [`docs/DESIGN_EVOLUTION.md`](docs/DESIGN_EVOLUTION.md) — how it got here: the incidents, rejected alternatives, and reasoning behind each addition.
- [`docs/LOW_LEVEL_DESIGN.md`](docs/LOW_LEVEL_DESIGN.md) — implementation-level detail, planted eval scenarios.
- [`docs/ERROR_HANDLING.md`](docs/ERROR_HANDLING.md) — retry/circuit-breaker/grounding-check design.
- [`docs/SECURITY.md`](docs/SECURITY.md) — tenant isolation, path-traversal guards, what's out of scope.
- [`docs/QUICKSTART_REVIEW.md`](docs/QUICKSTART_REVIEW.md) — the human-review demo, in full.
