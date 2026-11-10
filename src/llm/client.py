"""Claude Haiku 5.5 on the Message Batches API: key, token counts, batches, cost.

The key is read from ANTHROPIC_API_KEY or from .env and is never printed or logged.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

MODEL = "claude-haiku-5-5"
ROOT = Path(__file__).resolve().parents[2]

# Dollars per million tokens. Batch is half of standard, for prompts up to 100,000 tokens.
RATES = {
    "batch": {"input": 0.05, "output": 0.25},
    "standard": {"input": 0.10, "output": 0.50},
}

BATCH_TIMEOUT_MINUTES = 60
POLL_SECONDS = 30
BUDGET = 1.00


def load_key() -> None:
    """Put ANTHROPIC_API_KEY into the environment from .env if it is not already there."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip() == "ANTHROPIC_API_KEY" and value.strip():
                os.environ["ANTHROPIC_API_KEY"] = value.strip().strip("'\"")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("No ANTHROPIC_API_KEY in the environment or in .env. Save it there and rerun.")


def make_client():
    import anthropic

    load_key()
    return anthropic.Anthropic(max_retries=3)


def cost(input_tokens: int, output_tokens: int, tier: str = "batch") -> float:
    r = RATES[tier]
    return (input_tokens * r["input"] + output_tokens * r["output"]) / 1_000_000


def usage_cost(usage: dict, tier: str = "batch") -> float:
    return cost(usage.get("input_tokens", 0), usage.get("output_tokens", 0), tier)


def count_tokens(client, params: dict) -> int:
    """Free. Counts a request's input tokens with its system prompt and output_config."""
    keep = {k: params[k] for k in ("model", "system", "messages", "output_config") if k in params}
    return client.messages.count_tokens(**keep).input_tokens


def submit_batch(client, requests: list[dict]) -> str:
    """requests: [{"custom_id", "params"}]. Returns the batch id."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    batch = client.messages.batches.create(requests=[
        Request(custom_id=r["custom_id"], params=MessageCreateParamsNonStreaming(**r["params"]))
        for r in requests
    ])
    return batch.id


def wait_for_batch(client, batch_id: str, *, timeout_minutes: int = BATCH_TIMEOUT_MINUTES,
                   log=print) -> bool:
    """True when the batch ended. False after cancelling it for running past the timeout."""
    started = time.time()
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            return True
        if time.time() - started > timeout_minutes * 60:
            client.messages.batches.cancel(batch_id)
            log(f"batch {batch_id} ran past {timeout_minutes} minutes and was cancelled")
            return False
        time.sleep(POLL_SECONDS)


def collect_batch(client, batch_id: str) -> dict[str, dict]:
    """custom_id -> {status, usage, stop_reason, text, error, model}."""
    out: dict[str, dict] = {}
    for res in client.messages.batches.results(batch_id):
        row = {"status": res.result.type, "usage": None, "stop_reason": None, "text": None,
               "error": None}
        if res.result.type == "succeeded":
            msg = res.result.message
            row["usage"] = {"input_tokens": msg.usage.input_tokens,
                            "output_tokens": msg.usage.output_tokens}
            row["stop_reason"] = msg.stop_reason
            row["text"] = "".join(b.text for b in msg.content if b.type == "text")
        elif res.result.type == "errored":
            row["error"] = res.result.error.model_dump(mode="json")
        out[res.custom_id] = row
    return out


def run_standard(client, requests: list[dict]) -> dict[str, dict]:
    """Fallback for a stuck batch: the same requests on the standard API, at double the rates."""
    out: dict[str, dict] = {}
    for r in requests:
        try:
            msg = client.messages.create(**r["params"])
            out[r["custom_id"]] = {
                "status": "succeeded",
                "usage": {"input_tokens": msg.usage.input_tokens,
                          "output_tokens": msg.usage.output_tokens},
                "stop_reason": msg.stop_reason,
                "text": "".join(b.text for b in msg.content if b.type == "text"),
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 - recorded, not hidden
            out[r["custom_id"]] = {"status": "errored", "usage": None, "stop_reason": None,
                                   "text": None, "error": {"type": type(exc).__name__}}
    return out
