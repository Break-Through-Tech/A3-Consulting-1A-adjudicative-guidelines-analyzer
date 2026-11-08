"""Fine-tune one encoder with 13 guideline heads on leak-free text_features.

    python -m src.heads.train --model nlpaueb/legal-bert-base-uncased --max-length 512 --run-name legalbert-512
    python -m src.heads.train --model answerdotai/ModernBERT-base --max-length 2048 --run-name modernbert-2048
    python -m src.heads.train --model answerdotai/ModernBERT-base --max-length 4096 --run-name modernbert-4096

A smoke run adds ``--limit 200 --max-steps 20``. A run cut off mid-way picks up
from its last finished epoch with ``--resume``.

Everything goes to ``<output-dir>/<run-name>/``:
    train_log.csv       loss, learning rate, s/step, tokens/s, peak GPU memory
    val_log.csv         val loss and macro-F1 at every evaluation
    best_model.pt       weights of the best checkpoint by defensible-8 val macro-F1
    logits_val.npy      (n_val, 13, 3) logits of the best checkpoint
    logits_test.npy     (n_test, 13, 3)
    cases_<split>.csv   pool row and case_number for each logits row
    config.json         arguments, git commit, versions, data stats and timings
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.heads import data as heads_data
from src.heads.data import GUIDELINES, TEXT_COL
from src.heads.evaluate import head_metrics
from src.heads.model import GuidelineHeads, masked_head_loss

PROJECT_ROOT = heads_data.PROJECT_ROOT
CHECKPOINT = "checkpoint_last.pt"
BEST = "best_model.pt"
PROJECT_AFTER_STEPS = 50


def default_settings(model: str, max_length: int) -> tuple[int, int, float]:
    """Batch size, gradient accumulation and learning rate for each planned run."""
    if "modernbert" in model.lower():
        return (1, 16, 3e-5) if max_length > 2048 else (2, 8, 3e-5)
    return 16, 1, 2e-5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune 13 guideline heads on one encoder.")
    parser.add_argument("--model", required=True, help="Hugging Face model name")
    parser.add_argument("--max-length", type=int, required=True, help="tokens kept from the start of each text")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--warmup", type=float, default=0.1, help="share of steps spent warming up")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="use a seeded sample of this many cases from each split")
    parser.add_argument("--max-steps", type=int, help="stop after this many optimizer steps")
    parser.add_argument("--eval-every", type=int, default=0, help="also score val every N steps; 0 is epoch ends only")
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "runs"))
    parser.add_argument("--resume", action="store_true", help="continue from checkpoint_last.pt")
    parser.add_argument("--overwrite", action="store_true", help="replace a finished run of the same name")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--keep-checkpoint", action="store_true", help="keep checkpoint_last.pt after a finished run")
    args = parser.parse_args(argv)

    batch_size, grad_accum, lr = default_settings(args.model, args.max_length)
    args.batch_size = args.batch_size or batch_size
    args.grad_accum = args.grad_accum or grad_accum
    args.lr = args.lr or lr
    args.eval_batch_size = args.eval_batch_size or max(1, args.batch_size * 2)
    return args


# --- environment -------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_precision(choice: str, device: torch.device) -> str:
    """bf16 where the GPU runs it natively (compute capability 8 or more), else fp16."""
    if device.type != "cuda":
        return "fp32"
    if choice != "auto":
        return choice
    major, _ = torch.cuda.get_device_capability(device)
    return "bf16" if major >= 8 and torch.cuda.is_bf16_supported() else "fp16"


def autocast_factory(precision: str, device: torch.device):
    if precision == "fp32" or device.type != "cuda":
        return contextlib.nullcontext
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return lambda: torch.autocast(device_type="cuda", dtype=dtype)


def git_commit() -> dict[str, object]:
    def run(*cmd: str) -> str:
        try:
            out = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=20)
            return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    commit = run("git", "rev-parse", "HEAD") or "unknown"
    dirty = bool(run("git", "status", "--porcelain", "--untracked-files=no"))
    branch = run("git", "rev-parse", "--abbrev-ref", "HEAD") or "unknown"
    return {"commit": commit, "branch": branch, "dirty": dirty}


def versions(device: torch.device) -> dict[str, object]:
    import transformers

    info: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info["gpu"] = props.name
        info["gpu_memory_gb"] = round(props.total_memory / 2**30, 2)
        info["gpu_capability"] = ".".join(map(str, torch.cuda.get_device_capability(device)))
        info["gpu_count"] = torch.cuda.device_count()
    return info


def append_csv(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def format_seconds(seconds: float) -> str:
    hours, rest = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s"


# --- model pieces ------------------------------------------------------------


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lowered = name.lower()
        (no_decay if param.ndim < 2 or "norm" in lowered or "bias" in lowered else decay).append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused = all(p.is_cuda for p in decay + no_decay)
    return torch.optim.AdamW(groups, lr=lr, fused=fused) if fused else torch.optim.AdamW(groups, lr=lr)


def linear_schedule(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    def factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.no_grad()
def predict_logits(model, dataset, batch_size: int, device, autocast, collate) -> np.ndarray:
    """(n, 13, 3) float32 logits in dataset order, longest batches first."""
    model.eval()
    sampler = heads_data.LengthBucketBatchSampler(dataset.lengths, batch_size, shuffle=False)
    out = np.zeros((len(dataset), len(GUIDELINES), len(heads_data.CLASSES)), dtype=np.float32)
    for indices in sampler:
        batch = collate([dataset[i] for i in indices])
        with autocast():
            logits = model(
                batch["input_ids"].to(device, non_blocking=True),
                batch["attention_mask"].to(device, non_blocking=True),
            )
        out[batch["index"].numpy()] = logits.float().cpu().numpy()
    return out


def score_val(model, dataset, weights, args, device, autocast, collate) -> dict[str, object]:
    started = time.perf_counter()
    logits = predict_logits(model, dataset, args.eval_batch_size, device, autocast, collate)
    loss = float(
        masked_head_loss(torch.as_tensor(logits), torch.as_tensor(dataset.targets), torch.as_tensor(weights))
    )
    per_head, defensible = head_metrics(logits, dataset.targets)
    return {
        "logits": logits,
        "val_loss": loss,
        "defensible_macro_f1": defensible,
        "per_head_macro_f1": {code: metrics["macro_f1"] for code, metrics in per_head.items()},
        "seconds": time.perf_counter() - started,
    }


# --- main --------------------------------------------------------------------


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    run_dir = Path(args.output_dir) / args.run_name
    checkpoint_path = run_dir / CHECKPOINT
    if args.resume and not checkpoint_path.exists():
        raise SystemExit(f"--resume given but {checkpoint_path} does not exist")
    if not args.resume and run_dir.exists() and any(run_dir.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"{run_dir} already has files. Use --resume, --overwrite or a new --run-name.")
        for path in run_dir.iterdir():
            if path.is_file():
                path.unlink()
    run_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.allow_cpu:
        device = torch.device("cpu")
    else:
        raise SystemExit("CUDA is not available. Install a CUDA build of torch, or pass --allow-cpu.")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    precision = resolve_precision(args.precision, device)
    autocast = autocast_factory(precision, device)
    set_seed(args.seed)
    timings: dict[str, float] = {}
    run_started = time.perf_counter()

    # Data
    started = time.perf_counter()
    df = heads_data.load_pool()
    targets = heads_data.target_matrix(df)
    positions = {split: heads_data.split_positions(df, split, args.limit, args.seed) for split in heads_data.SPLITS}
    weights = heads_data.class_weights(targets[positions["train"]])
    timings["load_seconds"] = time.perf_counter() - started

    from transformers import AutoTokenizer

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    datasets = {}
    for split, pos in positions.items():
        ids = heads_data.tokenize(tokenizer, df[TEXT_COL].iloc[pos], args.max_length)
        datasets[split] = heads_data.EncodedCases(ids, targets[pos])
        pd.DataFrame({"row": pos, "case_number": df["case_number"].iloc[pos].to_numpy()}).to_csv(
            run_dir / f"cases_{split}.csv", index=False
        )
    timings["tokenize_seconds"] = time.perf_counter() - started
    data_stats = {
        split: {
            "cases": len(ds),
            "mean_tokens": float(ds.lengths.mean()),
            "share_truncated": float((ds.lengths >= args.max_length).mean()),
        }
        for split, ds in datasets.items()
    }
    print(
        f"Run {args.run_name}: {args.model} at {args.max_length} tokens, {precision}, "
        f"batch {args.batch_size} x {args.grad_accum}, lr {args.lr:g}, device {device}"
    )
    for split, stats in data_stats.items():
        print(
            f"  {split}: {stats['cases']:,} cases, mean {stats['mean_tokens']:.0f} tokens, "
            f"{stats['share_truncated']:.1%} truncated"
        )

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    collate = heads_data.make_collate(pad_id)
    train_set = datasets["train"]
    sampler = heads_data.LengthBucketBatchSampler(train_set.lengths, args.batch_size, shuffle=True, seed=args.seed)
    loader = torch.utils.data.DataLoader(
        train_set,
        batch_sampler=sampler,
        collate_fn=collate,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    # Model
    model = GuidelineHeads.from_pretrained(args.model, dropout=args.dropout)
    if not args.no_gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device)
    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    micro_per_epoch = len(sampler)
    steps_per_epoch = math.ceil(micro_per_epoch / args.grad_accum)
    total_steps = args.epochs * steps_per_epoch
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    epochs_to_run = math.ceil(total_steps / steps_per_epoch)
    warmup_steps = int(args.warmup * total_steps)
    scheduler = linear_schedule(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    weights_tensor = torch.as_tensor(weights, device=device)

    state = {
        "epoch": 0,
        "global_step": 0,
        "best_score": -math.inf,
        "best_step": None,
        "best_epoch": None,
        "train_seconds": 0.0,
        "eval_seconds": 0.0,
    }
    if args.resume:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint["args"]["model"] != args.model or checkpoint["args"]["max_length"] != args.max_length:
            raise SystemExit("The checkpoint was made with a different --model or --max-length.")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        state.update(checkpoint["state"])
        torch.set_rng_state(checkpoint["rng"]["torch"])
        if device.type == "cuda" and checkpoint["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state(checkpoint["rng"]["cuda"].cpu())
        print(f"Resumed after epoch {state['epoch']} at step {state['global_step']}")

    print(
        f"{steps_per_epoch:,} optimizer steps per epoch, {total_steps:,} in total, "
        f"{warmup_steps:,} warm-up steps"
    )

    def evaluate_and_keep(epoch: int, step: int) -> None:
        result = score_val(model, datasets["val"], weights, args, device, autocast, collate)
        state["eval_seconds"] += result["seconds"]
        improved = result["defensible_macro_f1"] > state["best_score"]
        if improved:
            state.update(best_score=result["defensible_macro_f1"], best_step=step, best_epoch=epoch)
            torch.save(model.state_dict(), run_dir / BEST)
        append_csv(
            run_dir / "val_log.csv",
            {
                "epoch": epoch,
                "step": step,
                "val_loss": result["val_loss"],
                "defensible_macro_f1": result["defensible_macro_f1"],
                "best": improved,
                "eval_seconds": round(result["seconds"], 1),
                **{f"macro_f1_{code}": result["per_head_macro_f1"].get(code) for code in GUIDELINES},
            },
        )
        print(
            f"  val at step {step}: loss {result['val_loss']:.4f}, defensible macro-F1 "
            f"{result['defensible_macro_f1']:.4f}{' (best)' if improved else ''}, {result['seconds']:.0f}s"
        )
        model.train()

    # Train
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    global_step = state["global_step"]

    def new_window() -> dict:
        return {"loss": 0.0, "micro": 0, "tokens": 0, "steps": 0, "elapsed": 0.0, "started": time.perf_counter()}

    window = new_window()
    train_started = time.perf_counter()
    steps_this_session = 0
    projected = False
    stop = global_step >= total_steps
    for epoch in range(state["epoch"], epochs_to_run):
        if stop:
            break
        sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_started = time.perf_counter()
        for micro, batch in enumerate(loader):
            window["tokens"] += int(batch["attention_mask"].sum())
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            batch_targets = batch["targets"].to(device, non_blocking=True)
            with autocast():
                logits = model(input_ids, attention_mask)
            loss = masked_head_loss(logits, batch_targets, weights_tensor)
            scaler.scale(loss / args.grad_accum).backward()
            window["loss"] += float(loss.detach())
            window["micro"] += 1

            last_micro = micro + 1 == micro_per_epoch
            if (micro + 1) % args.grad_accum and not last_micro:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            steps_this_session += 1
            window["steps"] += 1
            state["global_step"] = global_step

            if global_step % args.log_every == 0 or global_step == total_steps:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = window["elapsed"] + time.perf_counter() - window["started"]
                peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
                row = {
                    "epoch": epoch,
                    "step": global_step,
                    "loss": window["loss"] / max(1, window["micro"]),
                    "lr": scheduler.get_last_lr()[0],
                    "seconds_per_step": elapsed / max(1, window["steps"]),
                    "tokens_per_second": window["tokens"] / max(elapsed, 1e-9),
                    "peak_memory_gb": round(peak, 3),
                    "loss_scale": scaler.get_scale() if scaler.is_enabled() else 1.0,
                }
                append_csv(run_dir / "train_log.csv", row)
                print(
                    f"  step {global_step}/{total_steps} epoch {epoch} loss {row['loss']:.4f} "
                    f"lr {row['lr']:.2e} {row['seconds_per_step']:.2f}s/step "
                    f"{row['tokens_per_second']:,.0f} tok/s peak {peak:.2f} GB",
                    flush=True,
                )
                window = new_window()

            if not projected and steps_this_session == PROJECT_AFTER_STEPS:
                projected = True
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                per_step = (time.perf_counter() - train_started) / PROJECT_AFTER_STEPS
                state["seconds_per_step_first_50"] = per_step
                remaining = (total_steps - global_step) * per_step
                print(
                    f"  projected training time: {format_seconds(total_steps * per_step)} "
                    f"({per_step:.2f}s/step x {total_steps:,} steps), {format_seconds(remaining)} left, "
                    "plus val scoring",
                    flush=True,
                )

            if args.eval_every and global_step % args.eval_every == 0 and global_step < total_steps:
                state["train_seconds"] += time.perf_counter() - train_started
                window["elapsed"] += time.perf_counter() - window["started"]
                evaluate_and_keep(epoch, global_step)
                train_started = window["started"] = time.perf_counter()

            if global_step >= total_steps:
                stop = True
                break

        state["train_seconds"] += time.perf_counter() - train_started
        window["elapsed"] += time.perf_counter() - window["started"]
        print(f"Epoch {epoch} done in {format_seconds(time.perf_counter() - epoch_started)}", flush=True)
        evaluate_and_keep(epoch, global_step)
        state["epoch"] = epoch + 1
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "state": state,
                "args": vars(args),
                "rng": {
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state() if device.type == "cuda" else None,
                },
            },
            checkpoint_path,
        )
        train_started = window["started"] = time.perf_counter()

    # Best checkpoint: val and test logits
    started = time.perf_counter()
    model.load_state_dict(torch.load(run_dir / BEST, map_location=device, weights_only=True))
    for split in ("val", "test"):
        logits = predict_logits(model, datasets[split], args.eval_batch_size, device, autocast, collate)
        np.save(run_dir / f"logits_{split}.npy", logits)
    timings["final_predict_seconds"] = time.perf_counter() - started
    timings["train_seconds"] = state["train_seconds"]
    timings["val_scoring_seconds"] = state["eval_seconds"]
    timings["total_seconds"] = time.perf_counter() - run_started
    peak_memory = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0

    config = {
        "args": vars(args),
        "precision": precision,
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "git": git_commit(),
        "versions": versions(device),
        "data": data_stats,
        "class_weights": {code: weights[i].tolist() for i, code in enumerate(GUIDELINES)},
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "best": {
            "defensible_macro_f1_val_argmax": state["best_score"],
            "step": state["best_step"],
            "epoch": state["best_epoch"],
        },
        "timings": {key: round(value, 1) for key, value in timings.items()},
        "seconds_per_step_first_50": state.get("seconds_per_step_first_50"),
        "peak_memory_gb": round(peak_memory, 3),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if not args.keep_checkpoint and checkpoint_path.exists():
        checkpoint_path.unlink()
    print(
        f"Done in {format_seconds(timings['total_seconds'])}. Best val defensible macro-F1 "
        f"{state['best_score']:.4f} at step {state['best_step']}. Files in {run_dir}"
    )
    return run_dir


if __name__ == "__main__":
    main()
