#!/usr/bin/env python3
"""
Demo Console — Local Web UI
==============================
A small local server for demoing the pipeline end-to-end without touching
a terminal: generate a fresh fictional tenant (persona + small
hand-reviewable dataset, via generate_persona.py), run the full pipeline
against it, then browse the result as an email inbox, a synthesized
brief, and the dispatchable drafts Stage 3 produced.

Deliberately stdlib-only (http.server, no Flask) — consistent with the
rest of this codebase's dependency philosophy (see requirements.txt).
Every pipeline step is invoked exactly as a human would from the CLI
(subprocess, `sys.executable -m ...`), not re-implemented — this UI is a
thin viewer/trigger over the existing, already-tested entry points, not a
second code path that could drift from them. Results are then read back
off disk (ledgers, daily_brief.md) the same way a human inspecting the
output/ directory would.

Usage:
    python3 webui/server.py
    python3 webui/server.py --port 8765

Then open http://localhost:8765/
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

sys.path.insert(0, REPO_ROOT)

from digest.core import tenant_paths  # noqa: E402
from digest.core import citations  # noqa: E402
from digest.core.embeddings import embed_texts  # noqa: E402
from digest.core.persona import load_persona  # noqa: E402
from digest.parsers.calendar_parser import load_calendar  # noqa: E402
from digest.parsers.email_parser import load_inbox  # noqa: E402
from digest.parsers.notes_parser import load_notes  # noqa: E402

# Same allowlist tenant_paths.for_tenant itself enforces — validated again
# here, before this ever-so-slightly-more-exposed surface (a local HTTP
# server, even if only bound to localhost) constructs a subprocess argv or
# a filesystem path from client-supplied input. Belt-and-suspenders, not
# a second source of truth: tenant_paths.for_tenant still re-validates and
# is the actual enforcement point.
_TENANT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

_PROVIDER_CHOICES = {"ollama", "anthropic", "google", "openrouter", "deepseek"}

PIPELINE_TIMEOUT_SECONDS = 900  # generous — real LLM calls, several per step


class GenerationError(RuntimeError):
    pass


def _validate_tenant_id(tenant_id: str) -> str:
    if not isinstance(tenant_id, str) or not _TENANT_ID_RE.match(tenant_id):
        raise ValueError(f"Invalid tenant id: {tenant_id!r}")
    return tenant_id


def _validate_provider(provider: str) -> str:
    if provider not in _PROVIDER_CHOICES:
        raise ValueError(f"Invalid provider: {provider!r}")
    return provider


def _run_cli(args: list, timeout: int = PIPELINE_TIMEOUT_SECONDS) -> str:
    """Runs one pipeline step exactly as the CLI would (argv list, never
    shell=True — tenant_id/provider/model are validated before this is
    ever called, but passing a list rather than a shell string means even
    an unvalidated value couldn't inject a second command). Returns
    combined stdout+stderr; raises GenerationError with the tail of that
    output on a non-zero exit so the UI can surface *why* a step failed,
    not just that it did.
    """
    result = subprocess.run(
        [sys.executable] + args,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        raise GenerationError(output[-4000:] or f"{args[0]} exited {result.returncode} with no output")
    return output


def list_tenants() -> list[str]:
    tenants_dir = os.path.join(REPO_ROOT, "data", "tenants")
    if not os.path.isdir(tenants_dir):
        return []
    return sorted(
        d for d in os.listdir(tenants_dir)
        if os.path.isdir(os.path.join(tenants_dir, d)) and _TENANT_ID_RE.match(d)
    )


def generate_tenant(hint: str, provider: str, model: str) -> dict:
    tenant_id = f"demo-{int(time.time())}"
    _validate_tenant_id(tenant_id)
    _validate_provider(provider)

    args = ["generate_persona.py", "--tenant-id", tenant_id, "--provider", provider, "--model", model]
    if hint:
        args += ["--hint", hint]
    log = _run_cli(args)

    paths = tenant_paths.for_tenant(tenant_id)
    persona_text = load_persona(paths.persona_file)
    return {"tenant_id": tenant_id, "persona_text": persona_text, "log": log}


def run_pipeline(tenant_id: str, provider: str, model: str) -> dict:
    _validate_tenant_id(tenant_id)
    _validate_provider(provider)

    common = ["--tenant", tenant_id, "--provider", provider, "--model", model]
    steps = [
        ["-m", "digest.agents.triage_agent", "--map-only"] + common,
        ["-m", "digest.agents.calendar_agent", "--map-only"] + common,
        ["-m", "digest.agents.notes_agent", "--map-only"] + common,
        ["-m", "digest.orchestrator"] + common,
    ]
    logs = []
    for step in steps:
        logs.append(_run_cli(step))
    return {"log": "\n\n".join(logs)}


def get_persona(tenant_id: str) -> str:
    _validate_tenant_id(tenant_id)
    paths = tenant_paths.for_tenant(tenant_id)
    return load_persona(paths.persona_file)


def get_emails(tenant_id: str) -> list[dict]:
    _validate_tenant_id(tenant_id)
    paths = tenant_paths.for_tenant(tenant_id)
    emails = load_inbox(paths.inbox_dir)
    emails.sort(key=lambda e: e.get("date_key", ""))
    return [
        {
            "subject": e["subject"], "from": e["from"], "to": e["to"],
            "date": e["date"], "date_key": e["date_key"], "body": e["body"],
            # citations.load_citable_sources uses exactly this field as an
            # email's ref — matching it here is what lets a citation chip
            # find the right card via data-ref.
            "filename": e.get("filename", ""),
        }
        for e in emails
    ]


def get_calendar_events(tenant_id: str) -> list[dict]:
    _validate_tenant_id(tenant_id)
    paths = tenant_paths.for_tenant(tenant_id)
    events = load_calendar(paths.calendar_file)
    return [
        {
            "uid": e.get("uid", ""), "summary": e.get("summary", ""),
            "description": e.get("description", ""), "location": e.get("location", ""),
            "date_key": e.get("date_key", ""),
            "start": e["start"].isoformat() if e.get("start") else None,
            "end": e["end"].isoformat() if e.get("end") else None,
            "attendees": [a.get("name") or a.get("email", "") for a in e.get("attendees", [])],
        }
        for e in events
    ]


def get_notes(tenant_id: str) -> list[dict]:
    _validate_tenant_id(tenant_id)
    paths = tenant_paths.for_tenant(tenant_id)
    notes = load_notes(paths.notes_dir)
    notes.sort(key=lambda n: n.get("date_key", ""))
    return [
        {"note_id": n["note_id"], "title": n.get("title", ""), "date_key": n.get("date_key", ""), "body": n.get("body", "")}
        for n in notes
    ]


# assemble_brief (digest/orchestrator.py) always emits these exact `## `
# headers — parsing them back out of the saved markdown rather than
# re-running synthesis in-process, so this stays a pure read of whatever
# the last real pipeline run actually produced.
_SECTION_RE = re.compile(r"^## (.+)$", re.MULTILINE)
# Citation tags land after the closing "**" but before the line's "\n" — see
# _annotate_brief_with_citations. Group 2 (optional, non-greedy) is that
# trailing tag; parsed separately by _parse_citation_suffix.
_DISPATCH_ITEM_RE = re.compile(r"\*\*(.+?)\*\*(.*?)\n(.*?)(?=\n\*\*|\Z)", re.DOTALL)
_CITATION_SUFFIX_RE = re.compile(r"^\s*_\[source: (.+)\]_\s*$")
_CITATION_KIND_RE = re.compile(r"^(.*)\s\((embedding-matched|inferred)\)$")


def _annotate_brief_with_citations(tenant_id: str, brief_text: str) -> tuple[str, dict]:
    """Runs citations.cite_brief against this tenant's real source files
    (email/calendar/notes — tasks deliberately excluded, since the UI only
    has views for the first three) and returns the annotated text plus a
    {(source, ref): label} lookup for turning a bare ref like "0002.eml"
    into something readable client-side.

    embed_fn=embed_texts (not None) — best-effort local-embedding tier, on
    by default like citations.py's own CLI. cite_brief already degrades
    gracefully (prints a warning, keyword-only result) if Ollama isn't
    running, so this never raises just because the embedding tier is
    unavailable. llm=None — no LLM-judge tier here: this runs synchronously
    on every brief/drafts page load, and a paid LLM call per load isn't a
    tradeoff a demo viewer should pay implicitly. citations.py's CLI is
    still there for the full three-tier pass run explicitly.
    """
    paths = tenant_paths.for_tenant(tenant_id)
    sources = citations.load_citable_sources(paths.inbox_dir, paths.calendar_file, paths.notes_dir)
    annotated_text, _stats = citations.cite_brief(brief_text, sources, llm=None, embed_fn=embed_texts)
    label_lookup = {(s["source"], s["ref"]): s["label"] for s in sources}
    return annotated_text, label_lookup


def _parse_citation_suffix(suffix: str, label_lookup: dict) -> list[dict]:
    m = _CITATION_SUFFIX_RE.match(suffix)
    if not m:
        return []
    refs_blob = m.group(1)
    matched_via = "keyword"
    kind_m = _CITATION_KIND_RE.match(refs_blob)
    if kind_m:
        refs_blob, kind = kind_m.group(1), kind_m.group(2)
        matched_via = "embedding" if kind == "embedding-matched" else "llm"

    out = []
    for pair in refs_blob.split(", "):
        if ": " not in pair:
            continue
        source, ref = pair.split(": ", 1)
        out.append({
            "source": source, "ref": ref,
            "label": label_lookup.get((source, ref), ref),
            "matched_via": matched_via,
        })
    return out


def _split_citation_suffix(line: str) -> tuple[str, str]:
    """citations.cite_brief always appends the tag as ' _[source: ...]_' at
    the very end of an annotated line — a fixed, code-authored format, so a
    plain rfind is simpler and just as reliable as a regex here.
    """
    idx = line.rfind(" _[source: ")
    if idx == -1 or not line.endswith("]_"):
        return line, ""
    return line[:idx], line[idx + 1:]


def _bulletize(section_text: str, label_lookup: dict) -> list[dict]:
    bullets = []
    for raw_line in section_text.split("\n"):
        stripped = raw_line.strip()
        if not stripped:
            continue
        text, suffix = _split_citation_suffix(stripped)
        citations_list = _parse_citation_suffix(suffix, label_lookup) if suffix else []
        bullets.append({"text": text, "citations": citations_list})
    return bullets


def get_brief(tenant_id: str) -> dict:
    _validate_tenant_id(tenant_id)
    paths = tenant_paths.for_tenant(tenant_id)
    if not os.path.exists(paths.brief_file):
        return {"has_brief": False}

    with open(paths.brief_file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    text, label_lookup = _annotate_brief_with_citations(tenant_id, raw_text)

    headers = list(_SECTION_RE.finditer(text))
    sections = {}
    for i, m in enumerate(headers):
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        sections[m.group(1).strip()] = text[start:end].strip()

    freshness_notice = ""
    if text.lstrip().startswith("# Daily Brief"):
        preamble_end = headers[0].start() if headers else len(text)
        preamble = text[len("# Daily Brief"):preamble_end].strip()
        if preamble.startswith(">"):
            freshness_notice = preamble

    return {
        "has_brief": True,
        "freshness_notice": freshness_notice,
        "what_matters_today": _bulletize(sections.get("What Matters Today", ""), label_lookup),
        "what_might_be_missed": _bulletize(sections.get("What You Might Be Missing", ""), label_lookup),
    }


def get_drafts(tenant_id: str) -> list[dict]:
    _validate_tenant_id(tenant_id)
    paths = tenant_paths.for_tenant(tenant_id)
    if not os.path.exists(paths.brief_file):
        return []

    with open(paths.brief_file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    text, label_lookup = _annotate_brief_with_citations(tenant_id, raw_text)

    headers = list(_SECTION_RE.finditer(text))
    dispatch_text = ""
    for i, m in enumerate(headers):
        if m.group(1).strip() == "Quick Dispatches":
            start = m.end()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
            dispatch_text = text[start:end].strip()
            break
    if not dispatch_text:
        return []

    drafts = []
    for m in _DISPATCH_ITEM_RE.finditer(dispatch_text):
        summary = m.group(1).strip()
        suffix = m.group(2).strip()
        body = m.group(3).strip()
        citations_list = _parse_citation_suffix(suffix, label_lookup) if suffix else []
        if body.startswith(">"):
            draft_text = "\n".join(line.lstrip("> ").rstrip() for line in body.splitlines())
            drafted = True
        else:
            draft_text = ""
            drafted = False
        drafts.append({"summary": summary, "draft_text": draft_text, "drafted": drafted, "citations": citations_list})
    return drafts


# ─── HTTP layer ───────────────────────────────────────────────────────────


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write(f"{self.address_string()} - {fmt % args}\n")

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: str, content_type: str) -> None:
        try:
            with open(path, "rb") as f:
                body = f.read()
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/":
            self._send_file(os.path.join(STATIC_DIR, "index.html"), "text/html; charset=utf-8")
            return
        if path == "/app.js":
            self._send_file(os.path.join(STATIC_DIR, "app.js"), "application/javascript; charset=utf-8")
            return
        if path == "/style.css":
            self._send_file(os.path.join(STATIC_DIR, "style.css"), "text/css; charset=utf-8")
            return

        if path == "/api/tenants":
            self._send_json({"tenants": list_tenants()})
            return

        m = re.match(r"^/api/tenant/([^/]+)/(emails|brief|drafts|persona|calendar|notes)$", path)
        if m:
            tenant_id, resource = m.group(1), m.group(2)
            try:
                _validate_tenant_id(tenant_id)
                if resource == "emails":
                    self._send_json({"emails": get_emails(tenant_id)})
                elif resource == "brief":
                    self._send_json(get_brief(tenant_id))
                elif resource == "drafts":
                    self._send_json({"drafts": get_drafts(tenant_id)})
                elif resource == "persona":
                    self._send_json({"persona_text": get_persona(tenant_id)})
                elif resource == "calendar":
                    self._send_json({"events": get_calendar_events(tenant_id)})
                elif resource == "notes":
                    self._send_json({"notes": get_notes(tenant_id)})
            except ValueError as e:
                self._send_json({"error": str(e)}, status=400)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/api/generate":
            try:
                body = self._read_json_body()
                result = generate_tenant(
                    hint=(body.get("hint") or "").strip(),
                    provider=body.get("provider", "deepseek"),
                    model=body.get("model", "deepseek-chat"),
                )
                self._send_json(result)
            except (ValueError, GenerationError) as e:
                self._send_json({"error": str(e)}, status=400)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        if path == "/api/run":
            try:
                body = self._read_json_body()
                result = run_pipeline(
                    tenant_id=body.get("tenant_id", ""),
                    provider=body.get("provider", "deepseek"),
                    model=body.get("model", "deepseek-chat"),
                )
                self._send_json(result)
            except (ValueError, GenerationError) as e:
                self._send_json({"error": str(e)}, status=400)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        self.send_response(404)
        self.end_headers()


def parse_args():
    parser = argparse.ArgumentParser(description="Local demo-console web server for the digest pipeline")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main():
    args = parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Demo console running at http://127.0.0.1:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
