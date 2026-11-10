"""One encoder with 13 three-way guideline heads, and the masked loss.

Both Legal-BERT and ModernBERT use the same mean pooling and the same linear
heads, so the comparison between them is like for like.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.heads.data import CLASSES, GUIDELINES, IGNORE_INDEX


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average the token vectors that the attention mask keeps, in float32."""
    mask = attention_mask.unsqueeze(-1).to(torch.float32)
    total = (hidden.to(torch.float32) * mask).sum(dim=1)
    return total / mask.sum(dim=1).clamp(min=1.0)


class GuidelineHeads(nn.Module):
    """Encoder, mean pooling, dropout, then Linear(hidden, 13 * 3)."""

    def __init__(self, encoder: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.n_guidelines = len(GUIDELINES)
        self.n_classes = len(CLASSES)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder.config.hidden_size, self.n_guidelines * self.n_classes)

    @classmethod
    def from_pretrained(cls, name: str, dropout: float = 0.1) -> "GuidelineHeads":
        from transformers import AutoModel

        return cls(AutoModel.from_pretrained(name, attn_implementation="sdpa"), dropout=dropout)

    @classmethod
    def from_config(cls, encoder_config, dropout: float = 0.1) -> "GuidelineHeads":
        """A randomly initialised encoder, for tests that must not download."""
        from transformers import AutoModel

        return cls(AutoModel.from_config(encoder_config, attn_implementation="sdpa"), dropout=dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = mean_pool(hidden, attention_mask)
        logits = self.classifier(self.dropout(pooled).to(self.classifier.weight.dtype))
        return logits.float().view(-1, self.n_guidelines, self.n_classes)


def masked_head_loss(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Weighted cross-entropy per head, averaged over heads with a known target.

    logits  (batch, heads, classes)
    targets (batch, heads), with IGNORE_INDEX where the label is masked
    weights (heads, classes)

    Each head's loss is sum(w * nll) / sum(w) over its known targets, which is
    what ``F.cross_entropy(weight=...)`` computes for that head alone.
    """
    known = targets != IGNORE_INDEX
    safe_targets = targets.clamp(min=0)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    nll = -log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    heads = torch.arange(targets.shape[1], device=targets.device).expand_as(targets)
    w = weights.to(logits.device)[heads, safe_targets] * known
    weight_per_head = w.sum(dim=0)
    has_known = known.any(dim=0)
    if not has_known.any():
        return logits.sum() * 0.0
    per_head = (w * nll).sum(dim=0) / weight_per_head.clamp(min=1e-12)
    return per_head[has_known].mean()
