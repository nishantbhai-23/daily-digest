"""
In-House Summarization Quality Judge
========================================
Scores one MAP-phase (source, output) pair against four criteria —
groundedness, completeness, conciseness, coherence/tone — as a
supplement to digest_checks.py's existing deterministic keyword-coverage
checks, which verify a known signal wasn't dropped but say nothing about
whether the output is otherwise a *good* summary.

Built in-house rather than adopting DeepEval/RAGAS: both frameworks'
signature Faithfulness metric already works the same way this module's
groundedness dimension does (extract claims, verify each against the
source) — this project already has that pattern proven out, live, three
times over, in citations.py's own grounding. Adopting a third-party
framework here would mean trusting judge prompts this project can't fully
audit, for a much heavier dependency tree than this repo has ever taken
on, to reimplement something already built and tested.

Each dimension gets the treatment that actually fits it, not one uniform
"LLM judge, score everything":

- groundedness: the LLM lists claims + a verbatim supporting quote per
  claim; each quote is verified as a real substring of source_text in
  code (same check shape as citations.llm_match_sources' layer 2) —
  never trusted at face value.
- completeness: the LLM names items it believes are missing from
  output_text; each is cross-checked against output_text's own keywords
  — a claimed-missing item whose own keywords are already present is
  self-contradicting and gets flagged, not trusted. Genuine omissions
  pass through as advisory (there's nothing to verify an *absence*
  against, unlike a positive claim).
- conciseness: fully deterministic, no LLM at all — digest_checks.check_conciseness.
- coherence_tone: passed through directly, explicitly advisory/
  non-gating — the one dimension with no deterministic backstop, same
  treatment eval_synthesis_variants.py already gives latency/cost.

Usage:
    from digest.eval.quality_judge import judge_map_quality

    score = judge_map_quality(llm, source_text, output_text)
"""

from digest.core.cross_reference import title_keywords
from digest.core.ledger import validate_schema
from digest.core.llm import call_with_retry
from digest.eval.digest_checks import check_conciseness

QUALITY_JUDGE_SCHEMA = {"claims": ["text", "quote"], "missing": None}


def build_quality_judge_prompt() -> str:
    return (
        "You are evaluating how well OUTPUT captures SOURCE, along two "
        "dimensions that require judgment (a third, conciseness, is "
        "checked separately without you).\n\n"
        "GROUNDEDNESS: List every distinct factual claim made in OUTPUT. "
        "For each claim, give a short verbatim quote — copied character-"
        "for-character from SOURCE — that supports it. If a claim has no "
        "real supporting span in SOURCE, still list the claim, but leave "
        "quote as an empty string. Do not paraphrase or invent a quote.\n\n"
        "COMPLETENESS: List any essential action item, deadline, or "
        "decision that is clearly present in SOURCE but missing from "
        "OUTPUT. Only genuine omissions of essential information — not "
        "stylistic differences or minor detail.\n\n"
        "COHERENCE_TONE: Rate 1-5 how naturally OUTPUT reads on its own "
        "(no schema narration, no garbled grammar, no leftover "
        "formatting artifacts), with a one-line justification.\n\n"
        "Output strictly valid JSON:\n"
        '{"claims": [{"text": "...", "quote": "..."}], '
        '"missing": ["..."], '
        '"coherence_tone": {"score": 1, "notes": "..."}}\n\n'
        "Do not write any markdown wrappers, conversational pleasantries, "
        "or extra text."
    )


def judge_map_quality(llm, source_text: str, output_text: str) -> dict:
    """One batched LLM call producing groundedness/completeness/coherence
    judgments at once, plus a fully deterministic conciseness check —
    see module docstring for what each dimension actually verifies vs.
    takes on faith.

    Returns:
        {"groundedness": {"score": float, "unverified_claims": [str, ...]},
         "completeness": {"gaps": [str, ...], "contradicted_gaps": [str, ...]},
         "conciseness": {"ratio": float, "verdict": "ok"|"warning: ..."},
         "coherence_tone": {"score": int, "notes": str}}
    """
    prompt = build_quality_judge_prompt()
    context = f"SOURCE:\n{source_text}\n\n---\n\nOUTPUT:\n{output_text}"

    def _call():
        result = llm.chat_json(messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": context},
        ])
        errors = validate_schema(result, QUALITY_JUDGE_SCHEMA)
        if not isinstance(result.get("coherence_tone"), dict) or "score" not in result["coherence_tone"]:
            errors.append("Missing or invalid 'coherence_tone' object (must have a 'score' key)")
        if errors:
            raise ValueError(f"Invalid quality-judge output: {errors}")
        return result

    try:
        result = call_with_retry(_call)
    except Exception as e:
        print(f"   ⚠️  Quality judge failed after retries: {e}")
        return {
            "groundedness": {"score": None, "unverified_claims": []},
            "completeness": {"gaps": [], "contradicted_gaps": []},
            "conciseness": _score_conciseness(source_text, output_text),
            "coherence_tone": {"score": None, "notes": "judge call failed"},
        }

    return {
        "groundedness": _score_groundedness(result.get("claims", []), source_text),
        "completeness": _score_completeness(result.get("missing", []), output_text),
        "conciseness": _score_conciseness(source_text, output_text),
        "coherence_tone": result.get("coherence_tone", {"score": None, "notes": ""}),
    }


def _score_groundedness(claims: list, source_text: str) -> dict:
    """Verifies each claimed quote is a real substring of source_text
    (case-insensitive) — same check shape as citations.llm_match_sources'
    quote-existence layer. A claim with no quote, or a fabricated one,
    counts against the score and is named, not silently dropped.
    """
    if not claims:
        return {"score": 1.0, "unverified_claims": []}

    unverified = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        quote = (claim.get("quote") or "").strip()
        if not quote or quote.lower() not in source_text.lower():
            unverified.append(claim.get("text", "(no claim text given)"))

    verified_count = len(claims) - len(unverified)
    return {"score": verified_count / len(claims), "unverified_claims": unverified}


def _score_completeness(missing: list, output_text: str) -> dict:
    """A claimed-missing item whose own keywords already appear in
    output_text is self-contradicting — the judge said it's absent, but
    its own vocabulary is right there. Flagged as contradicted rather
    than trusted, same "verify what's checkable" posture as groundedness.
    Genuinely-missing items (no such contradiction) pass through as
    advisory gaps — there's nothing to verify an absence against.
    """
    gaps, contradicted = [], []
    output_lower = output_text.lower()
    for item in missing:
        if not isinstance(item, str) or not item.strip():
            continue
        keywords = title_keywords(item)
        if keywords and all(kw.lower() in output_lower for kw in keywords):
            contradicted.append(item)
        else:
            gaps.append(item)
    return {"gaps": gaps, "contradicted_gaps": contradicted}


def _score_conciseness(source_text: str, output_text: str) -> dict:
    warning = check_conciseness(output_text, source_text)
    ratio = (len(output_text.split()) / len(source_text.split())) if source_text.split() else 0.0
    return {"ratio": round(ratio, 2), "verdict": warning or "ok"}
