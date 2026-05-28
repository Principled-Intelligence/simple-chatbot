"""Super simple smoke test for the simple-chatbot API.

Assumes the server is already running. Start it in another terminal with
e.g. `uv run simple-chatbot serve --docs-dir ./docs`, then run:

    python smoke_test.py
    python smoke_test.py --base-url http://localhost:8000 --message "hi"

Exits 0 on success, non-zero on failure.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def _get_json(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--message", default="Say the single word: pong.")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    base = args.base_url.rstrip("/")

    try:
        print(f"→ GET {base}/v1/models")
        models = _get_json(f"{base}/v1/models", timeout=args.timeout)
        assert models.get("object") == "list", f"unexpected /v1/models payload: {models}"
        assert models.get("data"), "no models returned"
        model_id = models["data"][0]["id"]
        print(f"  ok — model_id={model_id}")

        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": args.message}],
        }
        print(f"→ POST {base}/v1/chat/completions (message={args.message!r})")
        resp = _post_json(f"{base}/v1/chat/completions", payload, timeout=args.timeout)

        assert resp.get("object") == "chat.completion", f"unexpected response: {resp}"
        choices = resp.get("choices") or []
        assert choices, "no choices in response"
        content = choices[0].get("message", {}).get("content")
        assert content, "empty assistant content"
        cid = resp.get("conversation_id", "<none>")
        print(f"  ok — conversation_id={cid}")
        print(f"  assistant: {content!r}")

        # Responses API probe
        resp_payload = {"input": args.message}
        print(f"→ POST {base}/v1/responses (input={args.message!r})")
        resp_body = _post_json(f"{base}/v1/responses", resp_payload, timeout=args.timeout)
        assert resp_body.get("object") == "response", f"unexpected /v1/responses response: {resp_body}"
        output_items = resp_body.get("output") or []
        message_items = [it for it in output_items if it.get("type") == "message"]
        assert message_items, "no message item in output"
        text = message_items[-1]["content"][0]["text"]
        assert text, "empty assistant text"
        print(f"  ok — response_id={resp_body['id']}, output_item_count={len(output_items)}")
        print(f"  assistant: {text!r}")

        print("PASS")
        return 0

    except (AssertionError, urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"FAIL — {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
