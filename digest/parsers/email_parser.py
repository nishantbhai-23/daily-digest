"""
Email Parser
=============
Parses .eml files into structured dicts using Python stdlib only.
Handles base64 and quoted-printable encoded bodies, extracts headers,
and derives thread IDs for conversation grouping.

Usage:
    from digest.parsers.email_parser import parse_eml, load_inbox, group_by_date

    email = parse_eml("data/inbox/0001.eml")
    all_emails = load_inbox("data/inbox/")
    daily_batches = group_by_date(all_emails)
"""

import email
import os
import re
from collections import defaultdict
from email import policy
from email.utils import parsedate_to_datetime


def parse_eml(filepath: str) -> dict:
    """Parse a single .eml file into a structured dict.

    Handles MIME decoding (base64, quoted-printable) and extracts
    all relevant headers. Derives a thread_id from the subject line
    by stripping Re:/Fwd: prefixes for conversation grouping.

    Args:
        filepath: Path to the .eml file.

    Returns:
        Dict with keys: subject, from, to, date, date_key, body,
        message_id, thread_id.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        msg = email.message_from_file(f, policy=policy.default)

    # Extract headers
    subject = str(msg.get("Subject", ""))
    from_addr = str(msg.get("From", ""))
    to_addr = str(msg.get("To", ""))
    message_id = str(msg.get("Message-ID", ""))
    date_str = str(msg.get("Date", ""))

    # Parse date into ISO format
    try:
        dt = parsedate_to_datetime(date_str)
        date_iso = dt.isoformat()
    except (ValueError, TypeError):
        date_iso = date_str

    # Decode body content
    raw_body = _decode_body(msg)
    body = clean_body(raw_body)

    # Derive thread ID for conversation grouping
    thread_id = _derive_thread_id(subject)

    return {
        "subject": subject,
        "from": from_addr,
        "to": to_addr,
        "date": date_iso,
        "date_key": date_iso[:10] if len(date_iso) >= 10 else "unknown",
        "body": body,
        "message_id": message_id,
        "thread_id": thread_id,
    }


def clean_body(body: str, max_chars: int = 1000) -> str:
    """Clean conversational noise, signatures, and limit overall length.

    This reduces total prompt token sizes while retaining critical action
    items, decisions, and deadlines.
    """
    if not body:
        return ""

    lines = body.splitlines()
    cleaned_lines = []

    # Common signature starts and footers
    sig_prefixes = (
        "--", "best,", "cheers,", "regards,", "thanks,", "warmly,", "sincerely,",
        "best regards,", "many thanks,", "slack", "pagerduty", "jira",
        "thanks for", "talk soon"
    )

    for i, line in enumerate(lines):
        # If we see a signature marker near the end of the email, we truncate
        stripped = line.strip().lower()
        if i >= len(lines) - 5 and any(stripped.startswith(prefix) for prefix in sig_prefixes):
            break
        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines).strip()

    # Truncate if it exceeds character budget
    if len(cleaned_text) > max_chars:
        cleaned_text = cleaned_text[:max_chars] + "... [body truncated]"

    return cleaned_text


def _decode_body(msg) -> str:
    """Decode the email body, handling base64 and quoted-printable encodings.

    For multipart messages, extracts the text/plain part. Falls back to
    empty string if no decodable content is found.
    """
    # Simple (non-multipart) messages
    if not msg.is_multipart():
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            try:
                return payload.decode(charset)
            except (UnicodeDecodeError, LookupError):
                return payload.decode("utf-8", errors="replace")
        return ""

    # Multipart: find the text/plain part
    for part in msg.walk():
        if part.get_content_type() == "text/plain":
            payload = part.get_payload(decode=True)
            if payload:
                charset = part.get_content_charset() or "utf-8"
                try:
                    return payload.decode(charset)
                except (UnicodeDecodeError, LookupError):
                    return payload.decode("utf-8", errors="replace")

    return ""


def _derive_thread_id(subject: str) -> str:
    """Derive a thread identifier by stripping reply/forward prefixes.

    Strips Re:, Fwd:, Fw: prefixes (case-insensitive, repeated) and
    normalizes whitespace to produce a stable thread key.

    Examples:
        "Re: Re: Project Aurora update" → "project aurora update"
        "[Atlas Migration] Weekly progress" → "[atlas migration] weekly progress"
    """
    # Strip Re:/Fwd:/Fw: prefixes (possibly repeated)
    cleaned = re.sub(
        r"^(?:(?:Re|Fwd|Fw)\s*:\s*)+", "", subject, flags=re.IGNORECASE
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned


def load_inbox(inbox_dir: str) -> list[dict]:
    """Load and parse all .eml files from the inbox directory.

    Args:
        inbox_dir: Path to directory containing .eml files.

    Returns:
        List of parsed email dicts, sorted chronologically by date.
    """
    emails = []

    if not os.path.exists(inbox_dir):
        print(f"❌ Error: Inbox directory '{inbox_dir}' not found.")
        return emails

    for filename in sorted(os.listdir(inbox_dir)):
        if not filename.endswith(".eml"):
            continue

        filepath = os.path.join(inbox_dir, filename)
        try:
            parsed = parse_eml(filepath)
            emails.append(parsed)
        except Exception as e:
            print(f"⚠️  Failed to parse {filename}: {e}")

    # Sort by date for chronological processing
    emails.sort(key=lambda e: e.get("date", ""))
    return emails


def group_by_date(emails: list[dict]) -> dict:
    """Group parsed emails by date (YYYY-MM-DD) for day-by-day MAP processing.

    Args:
        emails: List of parsed email dicts (must have 'date_key' field).

    Returns:
        Dict mapping date strings to lists of emails, sorted chronologically.
    """
    groups = defaultdict(list)
    for email_data in emails:
        date_key = email_data.get("date_key", "unknown")
        groups[date_key].append(email_data)

    # Return sorted by date
    return dict(sorted(groups.items()))
