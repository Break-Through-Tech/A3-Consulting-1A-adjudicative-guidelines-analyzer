"""Mean-pooled ModernBERT-4096 embeddings for every Guideline F pool case, for the mitigator head.

    python -m src.heads.embed --out runs/mitigator/mitigator_embeddings.npz
    python -m src.heads.embed --out x.npz --limit 3          # a quick check

The vector is the `pooled` tensor inside GuidelineHeads.forward: the mean of the
encoder's last hidden state over the unmasked tokens, before dropout and the
classifier. It is computed from `text_features`, the same input the heads read,
at 4096 tokens with the checkpoint from src.heads.infer.

The .npz holds three arrays, one row per case, in source_url order:

    source_url   strings
    split        strings, train, val or test
    embedding    float16, shape (n, 768)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from src.heads import data, infer

GUIDELINE = "F"


def f_cases(limit: int | None = None):
    """Guideline F rows of the pool across train, val and test, sorted by source_url."""
    from src.dataset import load_cases

    df = load_cases(pool=True, columns=["source_url", "guidelines", "split", data.TEXT_COL])
    df = df[df["guidelines"].apply(lambda g: GUIDELINE in list(g))]
    df = df.sort_values("source_url").reset_index(drop=True)
    if df["source_url"].duplicated().any():
        raise ValueError("source_url repeats in the Guideline F pool")
    return df.head(limit) if limit else df


def embed(texts: list[str], batch_size: int = 2, log=None) -> np.ndarray:
    """(n, hidden) float16 pooled vectors, in the order of `texts`."""
    import torch

    from src.heads.model import mean_pool

    model, tokenizer, device = infer._load(str(infer.checkpoint_path()))
    ids = data.tokenize(tokenizer, texts, infer.MAX_LENGTH)
    collate = data.make_collate(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    order = np.argsort([-len(row) for row in ids], kind="stable")      # longest first, little padding
    out = np.zeros((len(ids), model.classifier.in_features), dtype=np.float16)
    started = time.time()
    with torch.no_grad():
        for done, start in enumerate(range(0, len(order), batch_size)):
            chosen = order[start:start + batch_size]
            batch = collate([{"input_ids": ids[i], "targets": np.zeros(len(data.GUIDELINES), dtype=np.int64),
                              "index": int(i)} for i in chosen])
            ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" \
                else torch.autocast(device_type="cpu", enabled=False)
            with ctx:
                mask = batch["attention_mask"].to(device)
                hidden = model.encoder(input_ids=batch["input_ids"].to(device), attention_mask=mask).last_hidden_state
                pooled = mean_pool(hidden, mask)
            out[chosen] = pooled.float().cpu().numpy().astype(np.float16)
            if log and done % 100 == 0:
                log(f"{start + len(chosen)}/{len(order)} cases, {time.time() - started:.0f}s")
    return out


def write(out: Path, limit: int | None = None, batch_size: int = 2, log=print) -> int:
    started = time.time()
    df = f_cases(limit)
    log(f"{len(df)} Guideline F cases")
    vectors = embed(df[data.TEXT_COL].tolist(), batch_size=batch_size, log=log)
    if not np.isfinite(vectors.astype(np.float32)).all():
        raise ValueError("an embedding is not finite")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, source_url=df["source_url"].to_numpy(dtype=str), split=df["split"].to_numpy(dtype=str),
             embedding=vectors)
    log(f"wrote {out}: {len(df)} cases in {time.time() - started:.0f}s")
    return len(df)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, help="only the first N cases by source_url, for a check")
    ap.add_argument("--batch-size", type=int, default=2)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    if not infer.available():
        raise SystemExit(f"No checkpoint at {infer.checkpoint_path()}")
    write(args.out, args.limit, args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
