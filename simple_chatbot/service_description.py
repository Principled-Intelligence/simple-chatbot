# simple_chatbot/service_description.py
"""AI Service Descriptions (AISD) for the chatbot a server is serving.

Each fixture and the live RAG agent has a markdown AISD under `aisd/`, keyed by
fixture id (the live agent uses the reserved key `live-rag`). The server prints the
ACTIVE default chatbot's AISD at startup so it can be copy-pasted into Spectral.
The AISD describes capabilities only and never depends on the misbehavior config,
so good and evil resolve to identical text.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

LIVE_RAG_KEY = "live-rag"

_AISD_DIR = Path(__file__).parent / "aisd"

_LIVE_RAG_PLACEHOLDER = "(no service description available for the live RAG agent)"


def load_service_description(key: str) -> str | None:
    """Return the markdown AISD for a fixture id or LIVE_RAG_KEY, or None if the
    aisd/<key>.md file is absent."""
    path = _AISD_DIR / f"{key}.md"
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def resolve_active(config) -> tuple[str, str]:
    """Return (label, markdown) for the chatbot this server serves by default:
    config.default_fixture's AISD when set, else the live-rag AISD. Never reads the
    misbehavior config. On a missing file, falls back to the scenario's one-line
    description (or a placeholder) and warns — never raises."""
    key = config.default_fixture or LIVE_RAG_KEY
    text = load_service_description(key)
    if text is not None:
        return key, text

    logger.bind(key=key).warning("No AISD markdown file found; using fallback description")
    if key == LIVE_RAG_KEY:
        return key, (config.system_prompt or _LIVE_RAG_PLACEHOLDER)

    from simple_chatbot.scenario_registry import load_fixtures

    scenario = load_fixtures().get(key)
    fallback = (scenario.description if scenario else "") or (
        f"(no service description available for fixture {key!r})"
    )
    return key, fallback


def print_service_description(label: str, text: str) -> None:
    """Write a clearly-delimited AISD block to stdout (raw markdown, bypassing
    loguru) for clean copy-paste."""
    bar = "=" * 72
    print(f"\n{bar}\nAI SERVICE DESCRIPTION — {label}\n{bar}")
    print(text.rstrip("\n"))
    print(bar + "\n")
