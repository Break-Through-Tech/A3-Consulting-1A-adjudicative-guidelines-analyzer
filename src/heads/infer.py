"""Live ModernBERT-4096 inference for text the published outputs do not cover.

    from src.heads import infer
    infer.available()                     # is the checkpoint on disk
    out = infer.predict(text_features)    # the schema.load_head_outputs form

The checkpoint is runs/modernbert-4096/best_model.pt under the project root (600 MB,
not in git); MODERNBERT_CHECKPOINT points somewhere else. The encoder config and
tokenizer come from the base model named in config.json beside the checkpoint,
read from the Hugging Face cache. The model runs on the GPU when there is one,
under fp16 autocast as in training, and on the CPU otherwise. It is loaded once.

Probabilities are softmax(logits / T) per head, with the temperature and the
at-issue threshold from results/heads/modernbert-4096/calibration.json.
"""
from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_ENV = "MODERNBERT_CHECKPOINT"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "runs" / "modernbert-4096" / "best_model.pt"
CALIBRATION = PROJECT_ROOT / "results" / "heads" / "modernbert-4096" / "calibration.json"
LIVE_MODEL = "modernbert-4096-live"
MAX_LENGTH = 4096


def checkpoint_path() -> Path:
    override = os.environ.get(CHECKPOINT_ENV)
    return Path(override) if override else DEFAULT_CHECKPOINT


def available() -> bool:
    return checkpoint_path().is_file()


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=2)
def _load(path: str):
    """(model, tokenizer, device) for one checkpoint file, loaded once."""
    import torch
    from transformers import AutoConfig, AutoTokenizer

    from src.heads.model import GuidelineHeads

    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"No ModernBERT checkpoint at {checkpoint}. Put best_model.pt there or set {CHECKPOINT_ENV}.")
    config_path = checkpoint.with_name("config.json")
    base = "answerdotai/ModernBERT-base"
    if config_path.is_file():
        base = json.loads(config_path.read_text(encoding="utf-8"))["args"]["model"]
    device = _device()
    model = GuidelineHeads.from_config(AutoConfig.from_pretrained(base))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.to(device).eval()
    return model, AutoTokenizer.from_pretrained(base), device


def _calibration() -> tuple[dict, str]:
    raw = CALIBRATION.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()[:12]


def logits(texts: list[str], batch_size: int = 1):
    """(n, 13, 3) float32 numpy logits, before temperature."""
    import numpy as np
    import torch

    from src.heads import data

    model, tokenizer, device = _load(str(checkpoint_path()))
    ids = data.tokenize(tokenizer, texts, MAX_LENGTH)
    collate = data.make_collate(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    out = []
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            batch = collate([{"input_ids": row, "targets": np.zeros(len(data.GUIDELINES), dtype=np.int64),
                              "index": i} for i, row in enumerate(ids[start:start + batch_size])])
            ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" \
                else torch.autocast(device_type="cpu", enabled=False)
            with ctx:
                result = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            out.append(result.float().cpu().numpy())
    return np.concatenate(out)


def predict(text_features: str) -> dict:
    """Head outputs for one text, in `schema.load_head_outputs` form."""
    import numpy as np

    from src.heads import data

    calibration, digest = _calibration()
    raw = logits([text_features])[0]                       # (13, 3)
    guidelines = {}
    for i, code in enumerate(data.GUIDELINES):
        z = raw[i].astype(np.float64) / float(calibration[code]["temperature"])
        p = np.exp(z - z.max())
        p /= p.sum()
        guidelines[code] = {"p_not_at_issue": float(p[0]), "p_for": float(p[1]), "p_against": float(p[2]),
                            "at_issue_threshold": float(calibration[code]["at_issue_threshold"])}
    return {"model": LIVE_MODEL,
            "calibration": f"results/heads/modernbert-4096/calibration.json sha256:{digest}, live checkpoint "
                           f"{checkpoint_path().name}",
            "guidelines": guidelines}
