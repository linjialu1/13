"""S3 torch-side candidates — FIX1.

Fixes the three P0 blockers from admin/REVIEW_s3_protocol_final_20261017.md:

  P0-1  C1 mask was inverted: protocol requires freezing ONLY
        `encoder.rnn_cell.backbone`, training ff1/ff2/time_a/time_b +
        `temporal_attention` + `head` during Task1.
  P0-2  attention formula was changed to `tanh(scorer(seq))`; the accepted baseline is
        `scorer(tanh(seq))` (pipeline.py:39).  tanh and a linear map do not commute.
  P0-3  LN / POOL / GRU were never wired into the real forward path; setting an unused
        attribute changes nothing.  The classes below subclass the real model and
        override `representation_and_logits` (and replace `encoder` for GRU).

No numpy import here: this module must stay runnable in a torch-only process
(the numpy-MKL / torch-libiomp OMP conflict on this machine).
"""
from __future__ import annotations

import torch
from torch import nn

HIDDEN = 48
LN_EPS = 1e-5
JOINT_DIM = 56

# Task1 freeze prefix for C1 — protocol text, not derived from the implementation.
C1_FROZEN_PREFIX = "encoder.rnn_cell.backbone"


def _base_class():
    from trunk import acbm_interface as ai
    return ai.pipeline_module().TemporalAttentionSequenceClassifier


# ---------------------------------------------------------------- attention (baseline formula)
def attention_pool(seq: torch.Tensor, scorer: nn.Linear) -> torch.Tensor:
    """Baseline-exact pooling: softmax_t(scorer(tanh(seq))) weighted sum.

    Matches pipeline.py:39 (`scores = temporal_attention(torch.tanh(sequence))`).
    """
    scores = scorer(torch.tanh(seq)).squeeze(-1)      # [B, T]
    weights = torch.softmax(scores, dim=1)
    return torch.sum(weights.unsqueeze(-1) * seq, dim=1)


def mean_pool(seq: torch.Tensor) -> torch.Tensor:
    return seq.mean(dim=1)


# ---------------------------------------------------------------- C1 mask (protocol direction)
def c1_trainable_names(model: nn.Module) -> list[str]:
    """Task1-trainable set for C1: everything EXCEPT the backbone prefix."""
    return [name for name, _ in model.named_parameters()
            if not name.startswith(C1_FROZEN_PREFIX)]


def apply_c1_mask(model: nn.Module) -> dict:
    trainable = set(c1_trainable_names(model))
    if not trainable:
        raise RuntimeError("C1 mask left nothing trainable — check the model")
    rows = []
    for name, p in model.named_parameters():
        p.requires_grad_(name in trainable)
        rows.append({"name": name, "shape": list(p.shape),
                     "requires_grad": bool(p.requires_grad)})
    backbone = [r for r in rows if r["name"].startswith(C1_FROZEN_PREFIX)]
    return {"parameters": rows,
            "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "total_params": int(sum(p.numel() for p in model.parameters())),
            "n_trainable_tensors": len(trainable),
            "backbone_all_frozen": all(not r["requires_grad"] for r in backbone),
            "temporal_attention_trainable": all(
                r["requires_grad"] for r in rows if r["name"].startswith("temporal_attention.")),
            "head_trainable": all(r["requires_grad"] for r in rows
                                  if r["name"].startswith("head."))}


# ---------------------------------------------------------------- wired candidates
class _LNModel:
    """Mixin implementing LN wiring; instantiated via `make_candidate_class`."""


def make_candidate_classes():
    Base = _base_class()

    class LNModel(Base):
        """LayerNorm between encoder output and the baseline attention pooling."""

        def __init__(self, kind, input_size, hidden_size, n_classes):
            super().__init__(kind, input_size, hidden_size, n_classes)
            self.ln = nn.LayerNorm(hidden_size, eps=LN_EPS)

        def representation_and_logits(self, x):
            sequence, _ = self.encoder(x)
            sequence = self.ln(sequence)
            rep = attention_pool(sequence, self.temporal_attention)
            return rep, self.head(rep)

    class PoolModel(Base):
        """Time-mean pooling; temporal attention removed entirely."""

        def __init__(self, kind, input_size, hidden_size, n_classes):
            super().__init__(kind, input_size, hidden_size, n_classes)
            del self.temporal_attention

        def representation_and_logits(self, x):
            sequence, _ = self.encoder(x)
            rep = mean_pool(sequence)
            return rep, self.head(rep)

    class GRUEncoder(nn.Module):
        """GRU returning the FULL sequence with the same (sequence, state) contract."""

        def __init__(self, input_size, hidden_size):
            super().__init__()
            self.rnn = nn.GRU(input_size, hidden_size, num_layers=1,
                              batch_first=True, bidirectional=False)

        def forward(self, x):
            out, h = self.rnn(x)
            return out, h

    class GRUModel(Base):
        """GRU backbone + baseline attention pooling (formula unchanged)."""

        def __init__(self, kind, input_size, hidden_size, n_classes):
            super().__init__(kind, input_size, hidden_size, n_classes)
            self.encoder = GRUEncoder(input_size, hidden_size)

        def representation_and_logits(self, x):
            sequence, _ = self.encoder(x)
            rep = attention_pool(sequence, self.temporal_attention)
            return rep, self.head(rep)

    return {"LN_layernorm_injection": LNModel, "POOL_mean_pooling": PoolModel,
            "C2_gru_backbone": GRUModel}


SUPPORTED_WIRED = ("LN_layernorm_injection", "POOL_mean_pooling", "C2_gru_backbone")


def build_wired_candidate(candidate: str, input_channels: int, n_classes: int,
                          kind: str = "CfC"):
    """Build a REAL wired candidate model.  Unknown candidate -> error (no silent baseline)."""
    classes = make_candidate_classes()
    if candidate == "C1_backbone_freeze":
        from trunk import acbm_interface as ai
        model = ai.pipeline_module().TemporalAttentionSequenceClassifier(
            kind, input_channels, HIDDEN, n_classes)
        apply_c1_mask(model)
        return model
    if candidate not in classes:
        raise ValueError(f"unsupported wired candidate: {candidate}")
    return classes[candidate](kind, input_channels, HIDDEN, n_classes)


def parameter_table(model: nn.Module) -> list[dict]:
    return [{"name": n, "shape": list(p.shape), "requires_grad": bool(p.requires_grad),
             "numel": int(p.numel())} for n, p in model.named_parameters()]
