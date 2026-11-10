"""The pool, the 13-head target matrix, class weights and length-bucketed batches.

Targets come from ``targets_for_guideline`` in the TF-IDF baseline, so the
transformer heads see exactly the labels the baseline saw.
"""
from __future__ import annotations

import importlib.util
import math
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

from src import config
from src.dataset import load_cases

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = PROJECT_ROOT / "notebooks" / "04_guideline_classifier_baseline.py"

TEXT_COL = "text_features"
POOL_COLUMNS = [
    "case_number",
    "guidelines",
    "guideline_findings",
    "guideline_regime",
    TEXT_COL,
    "split",
]
SPLITS = ("train", "val", "test")
GUIDELINES = tuple(config.GUIDELINE_CODES)
IGNORE_INDEX = -100
MAX_CLASS_WEIGHT = 50.0


@lru_cache(maxsize=1)
def baseline() -> ModuleType:
    """The 04 TF-IDF baseline as a module. Its filename starts with a digit."""
    spec = importlib.util.spec_from_file_location("guideline_baseline_04", BASELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CLASSES = tuple(baseline().CLASSES)
CLASS_INDEX = {label: index for index, label in enumerate(CLASSES)}
DEFENSIBLE = tuple(code for code in GUIDELINES if baseline().GUIDELINE_TIERS[code] == "defensible")


def load_pool() -> pd.DataFrame:
    """The trainable pool with the frozen split, in load_cases row order."""
    df = load_cases(pool=True, columns=list(POOL_COLUMNS))
    if df["split"].isna().any():
        raise ValueError("Some trainable cases are missing frozen split assignments")
    unexpected = set(df["split"].unique()) - set(SPLITS)
    if unexpected:
        raise ValueError(f"Unexpected split values: {sorted(unexpected)}")
    return df


def target_matrix(df: pd.DataFrame) -> np.ndarray:
    """(n_cases, 13) int64: 0 NOT_AT_ISSUE, 1 FOR, 2 AGAINST, -100 masked."""
    targets = np.full((len(df), len(GUIDELINES)), IGNORE_INDEX, dtype=np.int64)
    for column, code in enumerate(GUIDELINES):
        labels = baseline().targets_for_guideline(df, code).to_numpy()
        known = np.array([isinstance(value, str) for value in labels])
        targets[known, column] = [CLASS_INDEX[value] for value in labels[known]]
    return targets


def head_labels(column: np.ndarray) -> np.ndarray:
    """Class names for one head's known integer targets."""
    return np.asarray(CLASSES, dtype=object)[column]


def class_weights(train_targets: np.ndarray, cap: float = MAX_CLASS_WEIGHT) -> np.ndarray:
    """(13, 3) float32 weights in sklearn's "balanced" form, capped.

    For each head, weight = n_known / (n_classes_present * class_count). A class
    absent from train keeps weight 1, which never matters since no target has it.
    """
    n_heads = train_targets.shape[1]
    weights = np.ones((n_heads, len(CLASSES)), dtype=np.float64)
    for head in range(n_heads):
        known = train_targets[:, head][train_targets[:, head] != IGNORE_INDEX]
        counts = np.bincount(known, minlength=len(CLASSES))
        present = counts > 0
        if present.any():
            weights[head, present] = len(known) / (present.sum() * counts[present])
    return np.minimum(weights, cap).astype(np.float32)


def split_positions(df: pd.DataFrame, split: str, limit: int | None = None, seed: int = 42) -> np.ndarray:
    """Row positions of one split, optionally a seeded sample of ``limit`` rows."""
    positions = np.flatnonzero(df["split"].eq(split).to_numpy())
    if limit is not None and 0 < limit < len(positions):
        rng = np.random.default_rng(seed)
        positions = np.sort(rng.choice(positions, size=limit, replace=False))
    return positions


def tokenize(tokenizer, texts, max_length: int, chunk_size: int = 2000) -> list[np.ndarray]:
    """Token ids per text, truncated to ``max_length`` with special tokens."""
    texts = list(texts)
    ids: list[np.ndarray] = []
    for start in range(0, len(texts), chunk_size):
        encoded = tokenizer(
            texts[start : start + chunk_size],
            truncation=True,
            max_length=max_length,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        ids.extend(np.asarray(row, dtype=np.int32) for row in encoded["input_ids"])
    return ids


class EncodedCases:
    """Token ids plus targets for one split, indexable by position."""

    def __init__(self, input_ids: list[np.ndarray], targets: np.ndarray):
        if len(input_ids) != len(targets):
            raise ValueError("input_ids and targets differ in length")
        self.input_ids = input_ids
        self.targets = targets
        self.lengths = np.array([len(ids) for ids in input_ids], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> dict:
        return {"input_ids": self.input_ids[index], "targets": self.targets[index], "index": index}


class LengthBucketBatchSampler:
    """Batches of similar length to cut padding.

    When shuffling, indices are shuffled, cut into chunks of
    ``batch_size * bucket_batches``, sorted by length inside each chunk, and
    the batches are shuffled again. Without shuffling, everything is sorted
    longest first, which is the cheapest order for prediction.
    """

    def __init__(
        self,
        lengths: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
        bucket_batches: int = 50,
    ):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.bucket_batches = bucket_batches
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _batches(self) -> list[np.ndarray]:
        n = len(self.lengths)
        if not self.shuffle:
            order = np.argsort(-self.lengths, kind="stable")
            return [order[i : i + self.batch_size] for i in range(0, n, self.batch_size)]
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(n)
        chunk = self.batch_size * self.bucket_batches
        batches: list[np.ndarray] = []
        for start in range(0, n, chunk):
            part = order[start : start + chunk]
            part = part[np.argsort(-self.lengths[part], kind="stable")]
            batches.extend(part[i : i + self.batch_size] for i in range(0, len(part), self.batch_size))
        return [batches[i] for i in rng.permutation(len(batches))]

    def __iter__(self):
        for batch in self._batches():
            yield batch.tolist()

    def __len__(self) -> int:
        n = len(self.lengths)
        if not self.shuffle:
            return math.ceil(n / self.batch_size)
        chunk = self.batch_size * self.bucket_batches
        full, rest = divmod(n, chunk)
        return full * self.bucket_batches + math.ceil(rest / self.batch_size)


def make_collate(pad_token_id: int, pad_to_multiple: int = 8):
    """Pad a list of examples to the longest one, rounded up to a multiple of 8."""
    import torch

    def collate(examples: list[dict]) -> dict:
        longest = max(len(example["input_ids"]) for example in examples)
        width = int(math.ceil(longest / pad_to_multiple) * pad_to_multiple)
        input_ids = torch.full((len(examples), width), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(examples), width), dtype=torch.long)
        for row, example in enumerate(examples):
            ids = torch.as_tensor(example["input_ids"], dtype=torch.long)
            input_ids[row, : len(ids)] = ids
            attention_mask[row, : len(ids)] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "targets": torch.as_tensor(np.stack([example["targets"] for example in examples])),
            "index": torch.as_tensor([example["index"] for example in examples]),
        }

    return collate
