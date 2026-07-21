# High-Level Design — Chief of Staff AI

## Purpose

A multi-tenant Chief-of-Staff system that reads four data sources — email,
calendar, notes, tasks — and produces a daily brief for whichever operator's
persona it's run for, answering four things, per the persona's own rubric:

1. What genuinely matters today.
2. What's being missed (forgotten threads, declined meetings, stale
   follow-ups, contradictions between sources).
3. What's dispatchable in under a minute (drafted, ready to send).
4. Where the data itself is stale, missing, or contradictory — stated
   honestly, not papered over.

Provider-agnostic across five LLM providers, and standard-library-only when using DeepSeek/OpenRouter (others require lightweight SDK installs). Every capability is invoked as an explicit CLI command by a human — there is no scheduler, daemon, or queue anywhere in this codebase.

## System shape

```
data/tenants/<id>/inbox/    ─▶ email_parser    ─▶ triage_agent.py    ─▶ email ledger    ─┐
data/tenants/<id>/calendar/ ─▶ calendar_parser ─▶ calendar_agent.py  ─▶ calendar ledger  ─┤
data/tenants/<id>/notes/    ─▶ notes_parser    ─▶ notes_agent.py     ─▶ notes ledger     ─┼─▶ orchestrator.py ─▶ daily_brief.md
data/tenants/<id>/tasks/    ─▶ tasks_parser    ─▶ tasks_signals.py   ─▶ live signals      ─┘        │
                                                                                                      ├─▶ citations.py         ─▶ daily_brief_cited.md
                                                                                                      └─▶ structured_digest.py ─▶ daily_brief_structured.json
```

Four independent per-source pipelines feed one orchestrator, run once per
tenant. `tenant_paths.for_tenant(id)` resolves every path above; `id="default"`
resolves to a flat `./data/`, `./output/` layout, any other id resolves under
`data/tenants/<id>/`, `output/tenants/<id>/`. None of the four per-source
pipelines reads another's data, and nothing in one tenant's run touches
another tenant's data.

`citations.py` and `structured_digest.py` are both optional, standalone
post-processing passes — run any time after the orchestrator has produced a
ledger, neither required for `daily_brief.md` to exist, neither modifying
anything the orchestrator itself owns.

## Pipeline stages

### MAP-REDUCE per source (email, calendar)

Each of `triage_agent.py` and `calendar_agent.py` processes its source
day-by-day: MAP extracts structured signals from one day's raw items into a
persistent, incrementally-resumable JSON ledger entry; REDUCE periodically
compacts older entries into a weekly rolling summary. A day already present
in the ledger is skipped on a later run — only new days are processed.

Every MAP call is preceded by deterministic, LLM-free computation the model
is never asked to re-derive: meeting overlap math
(`calendar_agent.compute_day_stats`), sender last-contact tracking
(`triage_agent.compute_sender_staleness`), today's actual date
(`ledger.format_today`, injected as a fact). The LLM's job is interpreting
what these facts mean, not computing them.

Whether MAP itself should see the persona (`use_persona_in_map`, default
`true`) is also evidence-based, not assumed — `eval_persona_map.py`
compares `build_map_system_prompt(use_persona=True)` against
`use_persona=False` (extraction-only framing: no priority field, generic
structural noise-filtering instead of persona-informed judgment) across
every source's `golden_scenarios.py` MAP scenarios. As of the last
comparison run (deepseek-chat, Tessera/Avery-CEO corpus), persona-aware
MAP scored 100% scenario coverage vs. 75% persona-free — two scenarios
that specifically require named-priority context
(`quiet_marcus_investor_thread`, `halberd_souring_signal`, both keyed to
names persona.md flags as high-priority) were under-extracted without
persona, since the model has no way to know a specific name matters
without it being told. `quality_judge.py`'s groundedness showed no
difference between the two (62% fully-grounded both ways) — persona
doesn't cost accuracy when present, it costs recall when removed. REDUCE
always gets full persona regardless of this toggle, so the question is
scoped narrowly to whether MAP-time extraction itself benefits, not
whether the operator ever sees persona-weighted output.

### Per-note MAP (notes)

`notes_agent.py` processes one note at a time rather than day-batching —
notes arrive too sparsely for daily batches to make sense. Deduped by
`note_id` via the ledger; adding one new note file only triggers one new MAP
call.

### Pre-structured input (tasks)

`tasks_signals.py` has no MAP phase and no ledger at all — `tasks.json` is
already typed, structured data (`priority`, `status`, `due_date` fields set
at creation), not prose to extract from. `compute_task_signals` is a pure,
deterministic function producing `overdue`/`due_soon`/`blocked`/`stalled`
buckets, recomputed fresh from the live file every run.

### Orchestrator (`orchestrator.py`)

Four stages, each with one job:

- **Stage 0** (no LLM by default): `cross_reference.CROSS_REFERENCE_VARIANTS`,
  selected via `--cross-reference-variant`. `lexical` (the default —
  `build_cross_reference_index`, keyword-matches flagged tasks against the
  other three ledgers) is purely deterministic. `embedding_assisted`
  (`build_cross_reference_index_embedding_assisted`) runs the identical
  lexical pass first, then adds a local-embedding rescue pass
  (`digest/core/embeddings.py`, via Ollama) for paraphrase matches the
  keyword pass misses — never removes a lexical match, only adds ones
  lexical missed. Not the default yet: this index directly seeds Stage 1's
  contradiction-detection candidate set, so a false match here has a
  bigger blast radius than a false positive anywhere else this codebase
  does keyword matching (it could feed a hallucinated contradiction about
  two things that were never actually the same task). Compared against the
  lexical baseline via `eval_cross_reference_variants.py` before any
  decision to change the default — as of the last comparison run against
  the arclight tenant, `embedding_assisted` found no additional mentions
  beyond lexical (the highest similarity among genuinely uncovered items
  was 0.524, well short of the 0.70 threshold — a real, evidence-based
  reason to stay conservative here, not an oversight). Both variants also
  run `check_data_freshness`.
- **Stage 1** (LLM, conditional): `detect_contradictions` — only runs when
  Stage 0 found a task mentioned in 2+ sources; skipped entirely otherwise.
  Every claimed contradiction is grounded before being trusted
  (`_ground_contradictions` verifies the claimed entity/sources were
  actually presented together) and the prompt is instructed to only assert
  what can be quoted directly from the given excerpts.
- **Stage 2** (LLM): unified synthesis, producing `what_matters_today`,
  `what_might_be_missed`, and a structured `dispatchable_items` list.
  Two interchangeable variants, selected via `--synthesis-variant`:
  `single_call` (one call for everything) and `staged` (narrative first,
  dispatchable items as a dependent second call). `check_priority_coverage`
  runs after either variant to verify every P0/P1 task from
  `tasks_signals.py` is mentioned somewhere in the output; logs a warning,
  never blocks. Two tiers: keyword matching first, then a local-embedding
  rescue (same `digest/core/embeddings.py`) for a task covered only via
  paraphrase — advisory-only, so the embedding tier is on by default here
  (unlike Stage 0's variant, there's no comparable blast-radius concern
  since this check never feeds another stage).
- **Stage 3** (LLM): draft generation for every dispatchable item, tone-
  locked to the persona, with `tenant_config.json`'s `never_draft_contacts`
  filtered out in code first — never left to prompt compliance.

Assembly (stitching Stage 2/3 output into `daily_brief.md`) is pure Python,
no LLM involved.

### Local embeddings (`digest/core/embeddings.py`)

Shared primitive behind every embedding-assisted matcher above
(citations' embedding tier, Stage 2's `check_priority_coverage` rescue,
Stage 0's `embedding_assisted` cross-reference variant): `embed_texts`
wraps Ollama's `nomic-embed-text` model, `cosine_similarity` and
`best_matches` are pure stdlib math — no numpy, no new dependency (reuses
the `ollama` package the Ollama chat provider already requires). Every
call site injects its own `embed_fn`, defaulting to the real function but
swappable for `None` (disables the tier) or a test double, and every
caller catches a failed embed call and falls back to whatever tier ran
before it rather than crashing. Each of the three call sites above
calibrates its own similarity threshold separately against real data
rather than sharing one value — the three comparison shapes (bullet vs.
full source document, task title vs. narrative chunk, task title vs. a
single extracted item) score differently even for equally genuine
matches, and the acceptable false-positive risk differs by how directly
each check's output feeds another stage (see Stage 0's writeup above for
the most consequential case).

### Citations (`digest/core/citations.py`)

Attaches a per-bullet source reference to an already-rendered
`daily_brief.md`, entirely after the fact. Splits the brief into bullets,
then matches each one against the day's real source items in three tiers,
cheapest first:

1. **Keyword matching** (`keyword_match_sources`) — free, deterministic,
   tried first. Uses `cross_reference.py`'s `title_keywords`/`find_mentions`
   primitives with prose-specific tuning (`_PROSE_STOPWORDS`, a match
   threshold that scales with bullet length, corpus-frequency filtering via
   `corpus_common_keywords`).
2. **Embedding matching** (`embedding_match_sources`) — bullets keyword
   matching can't place get compared against every source via local
   embeddings (`digest/core/embeddings.py`, Ollama's `nomic-embed-text`,
   cosine similarity ≥ 0.70). Catches paraphrase keyword overlap
   structurally can't — a real, live-observed case: a bullet synthesizing
   details from a 1:1 note scored 0.664 against its actual source, below
   an initial 0.78 guess but genuinely correct once recalibrated against
   real output. Also strengthens the LLM-judge layer below: its own
   "quote relevance" check now accepts either keyword overlap *or*
   embedding similarity, catching a real quote that supports a claim via
   paraphrase with zero literal word overlap.
3. **LLM-judge fallback** (`llm_match_sources`) for anything neither
   earlier tier could confidently place — one batched call, verified
   through four layers before a claimed match is trusted: the claimed
   `source_ref` must actually be one of the candidates shown; the model
   must supply a verbatim quote that is a real substring of that source's
   text; the quote must share a real keyword *or* clear the embedding
   threshold with the claim; and if the claim names a specific
   numbered/lettered instance (`"Press #2"`), the quote must name that
   same instance, not a different one sharing the same category noun.

A bullet with no confident match from any tier is left uncited rather
than guessed. The embedding tier is individually optional
(`--no-embeddings`) and degrades gracefully (prints a warning, falls
through to the LLM tier) if Ollama or the embedding model isn't
available. Output: `daily_brief_cited.md`, sibling to the original.

### Structured digest (`digest/core/structured_digest.py`)

A separate, optional pipeline producing a machine-readable, per-item JSON
representation of the digest instead of markdown prose. Reuses Stage 0/1
verbatim from `orchestrator.py` (same cross-reference index, contradiction
detection, shared context builder), then makes one new LLM call producing a
flat list of items, each explicitly typed:

```json
{"items": [{
  "id": "item-001", "type": "action|information|conflict",
  "title": "...", "summary": "...", "priority": "P0-P4",
  "due_date": "YYYY-MM-DD or null", "related_task_id": "... or null",
  "date_urgency": "overdue|due_today|due_soon|later|no_date",
  "source_refs": ["email:0002.eml", ...]
}]}
```

`date_urgency` is computed in Python from `due_date`, never trusted from the
model — same 7-day `due_soon` threshold `tasks_signals.py` uses. A
`priority_disagreement: true` flag is added (never auto-corrected) when the
model assigns a low priority (`P3`/`P4`) to an item whose `date_urgency` is
`overdue` or `due_today`. Citations attach directly to each item's `summary`
via the same `keyword_match_sources`/`llm_match_sources` functions citations
uses, with no markdown to parse. Output:
`daily_brief_structured.json`, sibling to `daily_brief.md`; the two never
interact.

Effort estimation ("how much work remains") is not part of this schema —
nothing in the data model tracks it today.

## Data model

**Ledger entry** (one per day/note, JSON, `{source}_rolling_ledger.json`):
```json
{
  "day": "2026-07-16",
  "email_count": 1,
  "stats": {"unique_senders": 1, "reply_count": 1, "new_thread_count": 0},
  "delta": {
    "deadlines": [], "decisions": [],
    "action_items": [{"description": "...", "priority": "P0"}],
    "thread_progressions": [{"thread": "...", "progression": "..."}]
  },
  "map_variant": "persona"
}
```
`stats` is deterministic; `delta` is the LLM's structured MAP output, schema
varies per source (see `docs/LOW_LEVEL_DESIGN.md`'s "Schemas reference").
`map_variant` records which MAP prompt/schema configuration produced this
entry. Optional `"hot": true` marks a `--hot-input`-sourced entry. Notes
entries add `"note_id"`.

**`tasks.json`**: a JSON array of task objects. Only `id` and `title` are
strictly required by the loaders; `status`, `priority`, `due_date`,
`created_at`, `description`, `tags`, `blocked_by`, `subtasks` are all
optional with sane defaults.

**`persona.md`**: freeform first-person markdown, always injected at
synthesis time. No enforced section structure, though `## People who
matter` and `## What I don't want surfaced` are what the prompts lean on
most.

**`tenant_config.json`** (optional — every key falls back to a default if
the file is missing):
```json
{
  "use_persona_in_map": true,
  "never_draft_contacts": [{"name": "...", "email": "..."}],
  "map_noise_filter": {"blocked_senders": [...], "blocked_domains": [...]}
}
```

**`system_config.json`** (fleet-wide, one file, not tenant-editable):
```json
{"max_qps_per_tenant": 2.0, "allowed_providers": [...], "circuit_breaker_threshold": 5}
```
Merged on top of every tenant's own config by `tenant_config.load_tenant_config`
— `SYSTEM_ENFORCED_KEYS` always come from this file, applied after the
tenant merge, so a tenant's own file can never override them.

## Multi-tenancy

`tenant_paths.for_tenant(tenant_id) -> TenantPaths` is the single place that
resolves where a tenant's data and output live. `tenant_id` is validated
against a conservative allowlist (`^[a-z0-9][a-z0-9_-]{0,63}$`) before any
path is constructed, closing a path-traversal vector.

`digest/run_fleet.py` runs several tenants concurrently via a
`ThreadPoolExecutor`, calling `orchestrator.run_for_tenant` directly in one
process — deliberately threads, not subprocesses, so `resilience.py`'s
circuit breaker and token bucket registries stay shared across every
tenant's calls.

## Provider abstraction

`digest/core/llm.py` defines a `BaseLLM` interface (`chat`, `chat_json`)
with five interchangeable providers — Ollama, Anthropic, Google, OpenRouter,
DeepSeek — behind a `create_llm` factory. Every script's `--provider`/
`--model` flags are identical. `.env` (gitignored) is loaded automatically
at import time via `digest/core/env.py`; a real environment variable always
overrides the file.

## Reliability layer

Every LLM call follows the same shape: `call_with_retry` (exponential
backoff), `validate_schema` (structural check on the parsed JSON before
accepting it), and a still-failing call surfaces as a visible, non-zero-exit
failure — never a silent skip.

- **Timeouts**: every provider sets an explicit request timeout
  (`DEFAULT_TIMEOUT_SECONDS`, 120s).
- **Error classification**: `TerminalLLMError` distinguishes errors that
  retrying can't fix (e.g. a 402 insufficient-balance response) from
  transient ones — terminal errors fail immediately instead of exhausting
  retries.
- **Circuit breaker** (`resilience.CircuitBreaker`): closed/open/half-open,
  keyed by provider+model, in-memory and shared within one process across
  every concurrent caller.
- **Grounding checks**: applied wherever an LLM's own claim about "did X
  happen" needs verifying before being trusted — Stage 1's contradiction
  claims, and citations' four-layer match verification described above.

## Data generation tooling

**`generate_data.py`**: generates the `default` tenant's ~30-day synthetic
corpus (email/calendar/notes/tasks) for one fixed persona (Avery Chen),
including deliberately planted scenarios tracked in `golden_scenarios.py`.

**`generate_persona.py`**: generates a fresh, small, fully-readable tenant
(5-10 emails, 2-3 calendar events, 2-3 notes, 2 tasks, across 3 consecutive
real days) for a new, randomly-invented or `--hint`-steered persona. One LLM
call produces structured JSON; deterministic Python renders it into actual
`.eml`/`.ics`/`.md`/`.json` files. Four checks run against the rendered
output before it's accepted, each re-verified through the real parsers
rather than trusted from the model's own claims, with one retry if any
fails:

1. `has_real_deep_work_conflict` — a genuine calendar-overlap conflict
   exists (via `calendar_agent.compute_day_stats`).
2. `count_exclusive_emails` — a majority of emails have content that
   appears nowhere else in the dataset.
3. `count_email_exclusive_notes` — at least one note has content that
   isn't also in an email (a note referencing its own calendar meeting is
   expected and doesn't count against this).
4. `find_calendar_date_inconsistencies` — every calendar event's own
   placement date agrees with what any email/note describing that same
   event says about it (matched via the same `keyword_match_sources`
   citations.py uses), using a weekday-name-to-date mapping built from the
   3 generated days.

`ANSWER_KEY.md` is also written per tenant — a human-reading aid describing
what a reviewer should expect to see at each of 3 incremental review steps
(`scripts/run_review_step.sh`). It is produced by the same generation call
as the raw data and is not used as an automated scoring signal anywhere in
the codebase; the actual automated eval (`golden_scenarios.py`) is
hand-authored and independent of any generation call.

`scripts/quickstart.sh` runs generation, the full pipeline, and citations in
one command for a fresh randomly-named tenant.

## Evaluation tooling

None of these run automatically. Most call a real model and cost real
provider budget; `eval_cross_reference_variants.py` is the one exception
(local embeddings only, no LLM call), noted below.

- **`golden_scenarios.py`**: a registry of real planted-data or directly-
  observed scenarios with known-correct expected signals, across MAP
  extraction, Stage-1 contradiction detection, priority calibration, and
  Stage-0 cross-referencing. `required_keywords` entries may be a plain
  string (must appear literally) or a nested list — an OR-set, at least
  one variant must appear — for concepts whose exact wording varies run
  to run.
- **`eval_map.py`**: runs MAP-phase extraction against
  `golden_scenarios.py` and scores it via `score_scenario`, a pure,
  LLM-free function (`digest_checks.check_keywords_present`/
  `extract_searchable_text`, leaf-string-based so schema key names never
  leak into the search corpus). Also runs `check_extraction_bloat` per
  entry — for email/calendar, the ceiling is computed dynamically
  (`dynamic_bloat_ceiling`) as `max(floor, batch_size * multiplier)`, so it
  scales with how much input a given day actually had rather than one flat
  number for every day; notes (MAP'd one note at a time, no natural batch
  size) uses a flat, real-corpus-calibrated ceiling. Optional `--llm-judge`
  flag additionally scores each scenario via `quality_judge.judge_map_quality`
  (see below) — a supplement to `score_scenario`'s coverage check, not a
  replacement.
- **`quality_judge.py`**: an in-house LLM judge scoring one (source,
  output) pair on four dimensions — groundedness (the judge lists claims
  with a supporting verbatim quote per claim; each quote is verified as a
  real substring of the source in code, same check shape as citations'
  own grounding, never trusted at face value), completeness (claimed-
  missing items are cross-checked against the output's own vocabulary — a
  claim that something's missing when its own keywords are already
  present is self-contradicting and gets flagged, not trusted),
  conciseness (fully deterministic, no LLM — `digest_checks.check_conciseness`,
  a length-ratio bound), and coherence/tone (the one dimension with no
  deterministic backstop, explicitly advisory/non-gating). Built in-house
  rather than adopting DeepEval/RAGAS — both frameworks' signature
  Faithfulness metric already works the same way this module's
  groundedness dimension does, and this project already had that pattern
  proven out, live, in citations.py's own grounding.
- **`eval_prompt_variants.py`** / **`eval_synthesis_variants.py`**: compare
  MAP-prompt and Stage-2-synthesis variants against each other across
  multiple trials, each run recorded to `eval_history/results.jsonl` with a
  content-addressed snapshot of the exact prompt text tested.
- **`eval_persona_map.py`**: compares `use_persona=True` vs. `False` MAP
  prompts across all three sources' `golden_scenarios.py` MAP scenarios —
  both keyword-coverage (`eval_map.score_scenario`, reused as-is) and,
  with `--llm-judge`, `quality_judge` groundedness. Two separate
  `eval_history.py` records per run (`map_persona_ablation` for coverage,
  `map_persona_quality_judge` for groundedness — keyed by persona mode,
  not provider/model, so they never collide with `eval_map.py`'s own
  `--llm-judge` records). See the MAP-REDUCE section above for the actual
  result this produced.
- **`eval_cross_reference_variants.py`**: compares Stage 0's `lexical` and
  `embedding_assisted` cross-reference variants against a tenant's real
  ledger data — the one eval script here that makes no LLM call at all
  (cross-referencing is keyword matching plus, optionally, local
  embeddings), so there's no provider budget concern and no run-to-run
  sampling variance to average over. Scores against
  `golden_scenarios.CROSS_REFERENCE_SCENARIOS` via
  `score_cross_reference_scenario` (a floor check — a variant finding a
  superset of the expected sources still passes) and surfaces total
  mention count and latency per variant, not scored — the tradeoff being
  measured is precision risk (a false cross-reference feeding a
  hallucinated contradiction), not $ cost.
- **`digest_checks.py`**: the shared pure-Python primitives several of the
  above scripts are built from — keyword presence/absence, category-scoped
  search, extraction-bloat bounds, schema-narration detection, conciseness
  ratio.

## Testing

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```
388 tests, fully offline, no LLM calls, no API keys needed — run in
milliseconds. This includes the embedding-assisted matchers above: every
test injects a fake/precomputed `embed_fn` rather than calling Ollama for
real, so the suite doesn't depend on it being installed or running.
Everything LLM-dependent lives in the eval scripts above instead, run
manually and explicitly.

## Known boundaries

Stated as current facts, not gaps awaiting a fix:

- **No scheduler.** Every script is a CLI entry point invoked by hand;
  `persona.md`'s "the digest runs at 6:00am sharp" describes the intended
  operator experience, not an implemented cron/daemon/queue.
- **No write-back or autonomous action.** Nothing in this system sends an
  email, creates or updates a task, or RSVPs to a calendar invite — Stage 3
  drafts text for a human to review and send; `structured_digest.py`'s
  `related_task_id` is a soft link for inspection, not a sync mechanism.
- **No database.** Multi-tenancy is filesystem-path resolution
  (`tenant_paths.py`); the two structural triggers that would justify a
  database — concurrent writers to the same shared record, or needing more
  than one process/machine — apply to neither of this system's current
  operating modes.
- **No distributed rate limiting.** `resilience.py`'s breaker/bucket state
  is in-process and in-memory; it doesn't survive a restart and isn't
  shared across machines. Sufficient as long as `run_fleet.py` stays one
  process.
- **`ledger.py`'s per-tenant lock covers individual load/save calls, not a
  full MAP-phase session** — two concurrent `run_map_phase` invocations for
  the same tenant can still race. Nothing that ships today triggers this,
  since nothing currently runs two MAP phases for the same tenant
  concurrently.
- **No prompt-injection fencing.** Email/note/calendar content is
  concatenated directly into MAP prompts with no delimiting; nothing stops
  content shaped like an instruction from appearing in that text. See
  `docs/SECURITY.md` for the full boundary and why the blast radius is
  currently small (nothing in this system auto-sends).
- **Every provider call leaves the process boundary.** Real or synthetic
  content reaches a third-party API for every provider, including
  non-US-domiciled ones. See `docs/SECURITY.md`'s data-handling section.
- **`system_config.json`'s `max_qps_per_tenant` is read but not enforced**
  — nothing in `orchestrator.py` or `run_fleet.py` constructs a
  `resilience.TokenBucket` from it yet.
- **`structured_digest.py` has no golden-scenario eval coverage** — nothing
  automated checks its `type`/`priority`/`date_urgency` classification
  accuracy today.
- **Every embedding-assisted matcher requires Ollama running locally with
  `nomic-embed-text` pulled** to do anything beyond its own keyword/
  lexical floor. All three call sites degrade gracefully (print a
  warning, fall back to the non-embedding result) if it isn't available
  — nothing crashes, but a tenant run without Ollama silently gets less
  recall than one with it, with no signal in the output that this
  happened beyond the printed warning.
- **`embedding_assisted` is not Stage 0's default** — `eval_cross_reference_variants.py`'s
  only comparison run so far (arclight) found no additional coverage
  beyond the lexical variant, so there's not yet a positive case for
  switching; this may look different on a tenant with more paraphrase-
  heavy source data.

## Further reading

- `docs/DESIGN_EVOLUTION.md` — how this architecture got here: the
  incidents, rejected alternatives, and reasoning behind each addition.
- `docs/LOW_LEVEL_DESIGN.md` — implementation-level detail, module-by-module.
- `docs/ERROR_HANDLING.md` — retry/circuit-breaker/grounding-check design.
- `docs/SECURITY.md` — tenant isolation, path-traversal guards, what's out
  of scope.
