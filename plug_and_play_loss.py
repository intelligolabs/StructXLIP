"""
StructXLIP — Plug-and-Play Loss Functions
==========================================
Three structure-aware training objectives from StructXLIP,
packaged as self-contained PyTorch modules.

Requirements
------------
- torch >= 1.10  (all losses)
- transformers   (LocalStructureCentricLoss only, for tokenizer)

Paper  : https://arxiv.org/abs/2602.20089
Project: https://eveleslie.github.io/structxlip-web/
Weights: https://huggingface.co/zanxii/StructXlip

Recommended starting point
---------------------------
Begin with StructureCentricAlignmentLoss — it is the simplest to integrate
and provides the most direct structural signal. Add the other two losses once
you have a baseline running.

    from plug_and_play_loss import StructureCentricAlignmentLoss

    loss_fn = StructureCentricAlignmentLoss()
    loss    = loss_clip + λ * loss_fn(scribble_emb, text_emb, has_struct, logit_scale)
"""

import re
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE over a (N, N) similarity matrix."""
    targets = torch.arange(similarity.shape[0], dtype=torch.long, device=similarity.device)
    return (
        F.cross_entropy(similarity,    targets) +
        F.cross_entropy(similarity.t(), targets)
    ) / 2.0


# ---------------------------------------------------------------------------
# 1. Structure-Centric Alignment  ← recommended starting point
# ---------------------------------------------------------------------------

class StructureCentricAlignmentLoss(nn.Module):
    """
    Structure-Centric Alignment loss.

    Symmetric contrastive alignment between global structural embeddings
    (e.g. from a scribble/edge map encoder) and their paired text embeddings.
    Only samples with a valid structural map (has_structure=True) contribute.

    **Recommended starting point** — requires only PyTorch, no model internals.

    Args:
        eps (float): Added before L2-normalisation for numerical stability.

    Inputs:
        scribble_embeds (B, D): global structural image embeddings
        text_embeds     (B, D): text embeddings paired with scribble_embeds
        has_structure   (B,)  : bool mask — True where a structural map exists
        logit_scale     scalar: learnable temperature, e.g. model.logit_scale.exp()

    Returns:
        scalar loss  (0.0 if no sample has a structural map)
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        scribble_embeds: torch.Tensor,
        text_embeds:     torch.Tensor,
        has_structure:   torch.Tensor,
        logit_scale:     torch.Tensor,
    ) -> torch.Tensor:
        mask = has_structure.bool()
        if not mask.any():
            return torch.tensor(0.0, device=scribble_embeds.device)

        img  = F.normalize(scribble_embeds[mask] + self.eps, dim=-1)
        text = F.normalize(text_embeds[mask]     + self.eps, dim=-1)
        return _clip_loss(logit_scale * (img @ text.t()))


# ---------------------------------------------------------------------------
# 2. RGB–Structure Consistency
# ---------------------------------------------------------------------------

class RGBStructureConsistencyLoss(nn.Module):
    """
    RGB–Structure Consistency loss.

    Cosine embedding loss that encourages paired RGB and structural embeddings
    to be close in the shared feature space.

    Requires only PyTorch.

    Inputs:
        image_embeds    (B, D): RGB image embeddings
        scribble_embeds (B, D): structural image embeddings (paired with image_embeds)
        has_structure   (B,)  : bool mask — True where a structural map exists

    Returns:
        scalar loss
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        image_embeds:    torch.Tensor,
        scribble_embeds: torch.Tensor,
        has_structure:   torch.Tensor,
    ) -> torch.Tensor:
        mask = has_structure.bool()
        if not mask.any():
            return torch.tensor(0.0, device=image_embeds.device)

        img   = F.normalize(image_embeds[mask]    + self.eps, dim=-1)
        struc = F.normalize(scribble_embeds[mask] + self.eps, dim=-1)
        ones  = torch.ones(mask.sum(), device=img.device)
        return F.cosine_embedding_loss(img, struc, ones)


# ---------------------------------------------------------------------------
# 3. Local Structure-Centric Alignment
# ---------------------------------------------------------------------------

def _adaptive_chunk_spans(
    caption: str,
    tokenizer,
    *,
    base_window: int = 3,
    stride:      int = 1,
    max_spans:   int = 64,
) -> list:
    """
    Slide a variable-width window over token spans of a caption.
    Window width adapts to sentence length. Returns list of
    (token_start, token_end_exclusive) tuples.
    """
    if not isinstance(caption, str) or not caption.strip():
        return []

    encoding = tokenizer(caption, return_offsets_mapping=True, add_special_tokens=True)
    words = list(re.finditer(r"\b\w+\b", caption))
    if not words:
        return []

    sentence_bounds, begin = [], 0
    for m in re.finditer(r"[.,;!?]+", caption):
        sentence_bounds.append((begin, m.end()))
        begin = m.end()
    if begin < len(caption):
        sentence_bounds.append((begin, len(caption)))

    spans, word_ptr = [], 0
    for sent_start, sent_end in sentence_bounds:
        start_wp = word_ptr
        while word_ptr < len(words) and words[word_ptr].start() < sent_end:
            word_ptr += 1
        end_wp   = word_ptr - 1
        sent_len = end_wp - start_wp + 1
        if sent_len <= 0:
            continue

        if   sent_len <= 4:  window = 2
        elif sent_len <= 10: window = base_window
        elif sent_len <= 20: window = base_window + 1
        elif sent_len <= 40: window = base_window + 2
        else:                window = base_window + 3

        for i in range(start_wp, end_wp - window + 2, stride):
            t_s = encoding.char_to_token(words[i].start())
            t_e = encoding.char_to_token(words[i + window - 1].end() - 1)
            if t_s is not None and t_e is not None and t_e >= t_s:
                spans.append((t_s, t_e + 1))
            if len(spans) >= max_spans:
                return spans

    return spans


class LocalStructureCentricLoss(nn.Module):
    """
    Local Structure-Centric Alignment loss.

    Aligns adaptive text-chunk embeddings with local structural segment
    embeddings via multi-positive InfoNCE. Valid structural segments from
    the same sample are positives; segments from other samples are negatives.

    Requires: the full StructXLIP model (for text_projection and logit_scale)
    and a HuggingFace tokenizer.

    Also returns a debug dict with diagnostics (margin, entropy, …).

    Args:
        chunk_base_window (int):   Base sliding-window width in words.
        chunk_stride      (int):   Stride of the sliding window.
        chunk_tau         (float): Temperature for the local contrastive loss.
        eps               (float): Normalisation epsilon.

    Inputs:
        model             : the full StructXLIP model
        text_tokens       (B, L, D): token-level text features
        captions          list[str]: raw caption strings (length B)
        edge_embeds_flat  (B*K, D): flattened edge/segment embeddings
        edge_valid_mask   (B, K)  : bool mask of valid segments per sample
        tokenizer         : HuggingFace tokenizer matching the text encoder

    Returns:
        (loss, debug)  — scalar tensor + diagnostics dict
    """

    def __init__(
        self,
        chunk_base_window: int   = 3,
        chunk_stride:      int   = 1,
        chunk_tau:         float = 0.07,
        eps:               float = 1e-6,
    ):
        super().__init__()
        self.chunk_base_window = chunk_base_window
        self.chunk_stride      = chunk_stride
        self.chunk_tau         = chunk_tau
        self.eps               = eps

    def forward(
        self,
        model,
        text_tokens:      torch.Tensor,
        captions:         Sequence[str],
        edge_embeds_flat: torch.Tensor,
        edge_valid_mask:  torch.Tensor,
        tokenizer,
    ):
        B, top_k = edge_valid_mask.shape
        neg_inf   = torch.finfo(edge_embeds_flat.dtype).min
        tau       = float(max(1e-6, self.chunk_tau))

        edge_all  = F.normalize(edge_embeds_flat + self.eps, dim=-1)
        valid_all = edge_valid_mask.reshape(-1).to(edge_all.device)

        debug_chunks, debug_pos, debug_ratio, debug_margin, debug_entropy = [], [], [], [], []
        total_loss, valid_items = torch.tensor(0.0, device=edge_all.device), 0

        for b in range(B):
            spans = _adaptive_chunk_spans(
                captions[b], tokenizer,
                base_window=self.chunk_base_window,
                stride=self.chunk_stride,
                max_spans=64,
            )
            if not spans:
                continue

            # Attention-weighted pooling over token spans → chunk embeddings
            chunk_embeddings = []
            tokens_b = text_tokens[b]
            for start, end in spans:
                if 0 <= start < end <= tokens_b.shape[0]:
                    with torch.no_grad():
                        w = F.softmax(tokens_b[start:end].norm(dim=-1), dim=0)
                    chunk_embeddings.append((tokens_b[start:end] * w[:, None]).sum(dim=0))
            if not chunk_embeddings:
                continue

            chunk_embeds = torch.stack(chunk_embeddings)
            chunk_embeds = model.text_model.final_layer_norm(chunk_embeds)
            chunk_embeds = model.text_projection(chunk_embeds)
            chunk_embeds = F.normalize(chunk_embeds + self.eps, dim=-1)   # (C, D)

            scale_eff = (model.logit_scale.exp() / tau).clamp(max=100.0)
            sim_all   = scale_eff * (chunk_embeds @ edge_all.t())          # (C, B*K)

            sim_den = sim_all.masked_fill(~valid_all.bool().unsqueeze(0), neg_inf)
            log_den = torch.logsumexp(sim_den, dim=1)

            pos_mask   = torch.zeros(B * top_k, dtype=torch.bool, device=edge_all.device)
            sample_pos = edge_valid_mask[b].to(edge_all.device)
            if not sample_pos.any():
                continue
            pos_mask[b * top_k: b * top_k + top_k] = sample_pos

            sim_pos = sim_all.masked_fill(~pos_mask.unsqueeze(0), neg_inf)
            log_pos = torch.logsumexp(sim_pos, dim=1)

            total_loss  += -(log_pos - log_den).mean()
            valid_items += 1

            with torch.no_grad():
                debug_chunks.append(float(chunk_embeds.shape[0]))
                debug_pos.append(float(sample_pos.sum().item()))
                debug_ratio.append(float(torch.exp(log_pos - log_den).mean().item()))
                neg_mask = valid_all & ~pos_mask
                if neg_mask.any():
                    top_p = sim_all[:, pos_mask].max(dim=1).values
                    top_n = sim_all[:, neg_mask].max(dim=1).values
                    debug_margin.append(float((top_p - top_n).mean().item()))
                log_probs = sim_den - log_den.unsqueeze(1)
                entropy   = -(torch.exp(log_probs) * log_probs).sum(dim=1).mean().item()
                debug_entropy.append(float(entropy))

        if valid_items > 0:
            total_loss = total_loss / valid_items

        def _avg(lst): return float(sum(lst) / len(lst)) if lst else 0.0

        debug = {
            "valid_items":    int(valid_items),
            "avg_chunks":     _avg(debug_chunks),
            "avg_pos":        _avg(debug_pos),
            "pos_mass_ratio": _avg(debug_ratio),
            "top1_margin":    _avg(debug_margin),
            "entropy":        _avg(debug_entropy),
        }
        return total_loss, debug


# ---------------------------------------------------------------------------
# Optional: loss weight schedule
# ---------------------------------------------------------------------------

def cosine_anneal_warm_decay(
    base_weight: float,
    epoch:       int,
    *,
    warm:        int   = 1,
    decay_start: int   = 3,
    decay_end:   int   = 8,
    floor:       float = 0.2,
) -> float:
    """
    Cosine annealing schedule with linear warm-up and a floor value.

    Phases:
      [0, warm)                  linear warm-up  → base_weight
      [warm, decay_start)        constant         = base_weight
      [decay_start, decay_end]   cosine decay    → floor * base_weight
      (decay_end, ∞)             constant         = floor * base_weight
    """
    if epoch < warm:
        factor = (epoch + 1) / max(1, warm)
    elif epoch < decay_start:
        factor = 1.0
    else:
        total    = max(1e-6, decay_end - decay_start)
        progress = min(float(epoch - decay_start), total)
        factor   = floor + (1 - floor) * 0.5 * (
            1 + torch.cos(torch.tensor(progress / total * torch.pi)).item()
        )
    return float(base_weight * factor)


# ---------------------------------------------------------------------------
# Smoke-test   →   python plug_and_play_loss.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    B, D = 8, 512

    scribble_emb = torch.randn(B, D)
    image_emb    = torch.randn(B, D)
    text_emb     = torch.randn(B, D)
    has_struct   = torch.tensor([True, True, False, True, True, False, True, True])
    logit_scale  = torch.tensor(4.6052).exp()   # exp(4.6) ≈ 100, typical CLIP init

    # 1. Structure-Centric Alignment (recommended starting point)
    sc_loss = StructureCentricAlignmentLoss()
    l1 = sc_loss(scribble_emb, text_emb, has_struct, logit_scale)
    print(f"Structure-Centric Alignment : {l1.item():.4f}")

    # 2. RGB–Structure Consistency
    rsc_loss = RGBStructureConsistencyLoss()
    l2 = rsc_loss(image_emb, scribble_emb, has_struct)
    print(f"RGB–Structure Consistency   : {l2.item():.4f}")

    # 3. Local Structure-Centric (skipped — requires real model + tokenizer)
    print("Local Structure-Centric     : skipped (requires model + tokenizer)")

    # Loss weight schedule
    print("\nLoss weight schedule over 10 epochs:")
    for ep in range(10):
        w = cosine_anneal_warm_decay(1.0, ep, warm=1, decay_start=3, decay_end=8)
        print(f"  epoch {ep:2d}  weight = {w:.3f}")
