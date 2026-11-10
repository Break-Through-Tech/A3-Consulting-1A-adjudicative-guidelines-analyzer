"""Tests for the transformer guideline heads in src/heads/.

They need no GPU and download nothing.

    python tests/test_heads.py
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from _harness import parametrize, run_module

from src.heads import data as heads_data
from src.heads.data import CLASSES, GUIDELINES, IGNORE_INDEX, baseline
from src.heads.evaluate import fit_temperature, nll, reliability, softmax
from src.heads.model import GuidelineHeads, masked_head_loss, mean_pool

SLICE = 3000


@lru_cache(maxsize=1)
def pool_slice() -> pd.DataFrame:
    return heads_data.load_pool().iloc[:SLICE]


# --- targets -----------------------------------------------------------------


@parametrize("code", list(GUIDELINES))
def test_target_matrix_matches_baseline(code):
    df = pool_slice()
    column = GUIDELINES.index(code)
    matrix = heads_data.target_matrix(df)[:, column]
    expected = baseline().targets_for_guideline(df, code)

    masked = expected.isna().to_numpy()
    assert (matrix == IGNORE_INDEX).sum() == masked.sum(), code
    assert np.array_equal(matrix == IGNORE_INDEX, masked), code
    decoded = heads_data.head_labels(matrix[~masked])
    assert list(decoded) == expected[~masked].tolist(), code


def test_target_matrix_masks_some_cases():
    # Listed guidelines with no FOR / AGAINST finding exist in the pool and must be masked.
    matrix = heads_data.target_matrix(pool_slice())
    assert (matrix == IGNORE_INDEX).any()
    assert set(np.unique(matrix)) <= {IGNORE_INDEX, 0, 1, 2}


def test_class_weights_balanced_and_capped():
    targets = np.full((100, 2), IGNORE_INDEX, dtype=np.int64)
    targets[:90, 0] = 0
    targets[90:99, 0] = 1
    targets[99, 0] = 2
    targets[:50, 1] = 0
    targets[50:, 1] = 1
    weights = heads_data.class_weights(targets, cap=50.0)
    # head 0: 100 known, 3 classes present: 100 / (3 * count)
    assert np.allclose(weights[0], [100 / 270, 100 / 27, 100 / 3])
    # head 1: 100 known, 2 classes present; AGAINST is absent and stays at 1
    assert np.allclose(weights[1], [1.0, 1.0, 1.0])
    capped = heads_data.class_weights(targets, cap=10.0)
    assert np.isclose(capped[0, 2], 10.0)


def test_length_bucket_sampler_covers_every_case_once():
    lengths = np.random.default_rng(0).integers(10, 500, size=1003)
    sampler = heads_data.LengthBucketBatchSampler(lengths, batch_size=16, shuffle=True, bucket_batches=5)
    batches = list(sampler)
    assert len(batches) == len(sampler)
    seen = np.concatenate(batches)
    assert sorted(seen.tolist()) == list(range(len(lengths)))
    sampler.set_epoch(1)
    assert [b for b in sampler] != batches
    ordered = list(heads_data.LengthBucketBatchSampler(lengths, batch_size=16, shuffle=False))
    flat = np.concatenate(ordered)
    assert np.all(np.diff(lengths[flat]) <= 0)


# --- loss and pooling --------------------------------------------------------


def test_masked_loss_ignores_masked_targets_and_weights_classes():
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(6, len(GUIDELINES), len(CLASSES), generator=generator)
    targets = torch.randint(0, len(CLASSES), (6, len(GUIDELINES)), generator=generator)
    targets[:2, 0] = IGNORE_INDEX
    targets[:, 3] = IGNORE_INDEX  # a head with nothing known is left out of the mean
    weights = torch.rand(len(GUIDELINES), len(CLASSES), generator=generator) * 5 + 0.1

    per_head = []
    for head in range(len(GUIDELINES)):
        known = targets[:, head] != IGNORE_INDEX
        if known.any():
            per_head.append(F.cross_entropy(logits[known, head], targets[known, head], weight=weights[head]))
    expected = torch.stack(per_head).mean()
    got = masked_head_loss(logits, targets, weights)
    assert torch.allclose(got, expected, atol=1e-6), (got, expected)

    changed = logits.clone()
    changed[:2, 0] = torch.tensor([50.0, -50.0, 3.0])
    changed[:, 3] = 99.0
    assert torch.allclose(masked_head_loss(changed, targets, weights), got, atol=1e-6)


def test_masked_loss_weight_changes_the_loss():
    logits = torch.zeros(2, len(GUIDELINES), len(CLASSES))
    logits[0, :, 0] = 2.0
    targets = torch.full((2, len(GUIDELINES)), IGNORE_INDEX)
    targets[0, 0] = 0  # confident and right
    targets[1, 0] = 2  # unsure, so a higher loss
    flat = torch.ones(len(GUIDELINES), len(CLASSES))
    heavy = flat.clone()
    heavy[0, 2] = 10.0
    assert masked_head_loss(logits, targets, heavy) > masked_head_loss(logits, targets, flat)


def test_mean_pool_ignores_padding():
    hidden = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    padded = hidden.clone()
    padded[0, 2:] = 1e6
    pooled = mean_pool(padded, mask)
    assert torch.allclose(pooled[0], hidden[0, :2].mean(dim=0))
    assert torch.allclose(pooled[1], hidden[1].mean(dim=0))


def test_tiny_bert_forward_shape_and_backward():
    from transformers import BertConfig

    config = BertConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    model = GuidelineHeads.from_config(config)
    input_ids = torch.randint(5, 100, (3, 10))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[0, 6:] = 0
    logits = model(input_ids, attention_mask)
    assert logits.shape == (3, len(GUIDELINES), len(CLASSES))
    targets = torch.zeros(3, len(GUIDELINES), dtype=torch.long)
    masked_head_loss(logits, targets, torch.ones(len(GUIDELINES), len(CLASSES))).backward()
    assert model.classifier.weight.grad is not None


def test_collate_pads_to_multiple_of_eight():
    collate = heads_data.make_collate(pad_token_id=0)
    examples = [
        {"input_ids": np.array([5, 6, 7]), "targets": np.zeros(13, dtype=np.int64), "index": 0},
        {"input_ids": np.array([5] * 10), "targets": np.ones(13, dtype=np.int64), "index": 1},
    ]
    batch = collate(examples)
    assert batch["input_ids"].shape == (2, 16)
    assert batch["attention_mask"][0].tolist() == [1, 1, 1] + [0] * 13
    assert batch["targets"].shape == (2, 13)


# --- calibration and decoding ------------------------------------------------


def test_temperature_fit_lowers_nll_on_overconfident_logits():
    rng = np.random.default_rng(0)
    n = 2000
    true_logits = rng.normal(size=(n, 3))
    probabilities = softmax(true_logits)
    targets = np.array([rng.choice(3, p=p) for p in probabilities])
    overconfident = true_logits * 4.0
    temperature = fit_temperature(overconfident, targets)
    assert 3.0 < temperature < 5.5, temperature
    assert nll(overconfident, targets, temperature) < nll(overconfident, targets)

    ece_before, _ = reliability(softmax(overconfident), targets)
    ece_after, bins = reliability(softmax(overconfident / temperature), targets)
    assert ece_after < ece_before
    assert bins["count"].sum() == n


def test_fit_temperature_handles_no_targets():
    assert fit_temperature(np.zeros((0, 3)), np.zeros(0, dtype=np.int64)) == 1.0


FIXED_PROBABILITIES = np.array(
    [
        [0.90, 0.05, 0.05],
        [0.40, 0.35, 0.25],
        [0.20, 0.30, 0.50],
        [0.10, 0.60, 0.30],
        [0.70, 0.20, 0.10],
        [0.05, 0.15, 0.80],
    ]
)
FIXED_LABELS = pd.Series(["NOT_AT_ISSUE", "FOR", "AGAINST", "FOR", "NOT_AT_ISSUE", "AGAINST"])


def test_threshold_and_decode_match_baseline_on_fixed_arrays():
    threshold = baseline().choose_at_issue_threshold(FIXED_LABELS, FIXED_PROBABILITIES)
    # At-issue scores are 0.10 0.60 0.80 0.90 0.30 0.95; 0.60 is the highest
    # threshold that keeps every at-issue case.
    assert np.isclose(threshold, 0.60)
    decoded = baseline().decode_predictions(FIXED_PROBABILITIES, threshold)
    assert decoded.tolist() == ["NOT_AT_ISSUE", "FOR", "AGAINST", "FOR", "NOT_AT_ISSUE", "AGAINST"]

    # The evaluate path feeds softmax(logits / T). With T = 1 and log-probability
    # logits it must reproduce the baseline answers exactly.
    via_logits = softmax(np.log(FIXED_PROBABILITIES))
    assert np.allclose(via_logits, FIXED_PROBABILITIES)
    via_threshold = baseline().choose_at_issue_threshold(FIXED_LABELS, via_logits)
    assert np.isclose(via_threshold, threshold)
    assert baseline().decode_predictions(via_logits, via_threshold).tolist() == decoded.tolist()


def test_decode_prefers_for_on_ties_like_baseline():
    tie = np.array([[0.2, 0.4, 0.4]])
    assert baseline().decode_predictions(tie, 0.5).tolist() == ["FOR"]
    assert baseline().decode_predictions(tie, 0.9).tolist() == ["NOT_AT_ISSUE"]


if __name__ == "__main__":
    raise SystemExit(run_module(globals(), "test_heads"))
