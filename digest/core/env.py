"""
Local Env Loading
===================
Repo-scoped provider API keys, loaded from a gitignored `.env` file at the
repo root — not a global shell env var, not typed inline in commands.
Closes a gap docs/SECURITY.md already flagged: exporting keys directly in
shell commands (this session's actual practice before this module existed)
leaves them in shell history and process listings; a `.env` file, gitignored,
is the minimum real improvement over that without introducing a secrets
vault this project doesn't need at its current scale.

Stdlib-only, deliberately — consistent with this codebase's dependency
philosophy (see HLD Decision 6 and the "what was avoided" section). No
`python-dotenv` package; `.env` here is a plain `KEY=VALUE` file, one per
line, `#`-comments and blank lines ignored.

Uses `os.environ.setdefault`, not assignment — a real environment variable
set some other way (CI secrets, a deploy environment) always wins over
`.env`, so this is safe to load unconditionally without risk of silently
overriding an intentionally-set value.

Usage:
    from digest.core.env import load_dotenv
    load_dotenv()  # call once, early — llm.py does this at import time
"""

import os

DOTENV_FILE = ".env"


def load_dotenv(path: str = DOTENV_FILE) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ, if present.

    Missing file is not an error — most environments (CI, a real deployment)
    won't have one and are expected to set real environment variables
    instead.
    """
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())
