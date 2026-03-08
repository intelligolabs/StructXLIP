import re
from typing import Sequence

import torch
import torch.nn.functional as F


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    targets = torch.arange(similarity.shape[0], dtype=torch.long, device=similarity.device)
    return (F.cross_entropy(similarity, targets) + F.cross_entropy(similarity.t(), targets)) / 2.0


def cosine_anneal_warm_decay(
    base_weight: float,
    epoch: int,
    *,
    warm: int = 1,
    decay_start: int = 3,
    decay_end: int = 8,
    floor: float = 0.2,
) -> float:
    if epoch < warm:
        factor = (epoch + 1) / max(1, warm)
    elif epoch < decay_start:
        factor = 1.0
    else:
        total = max(1e-6, decay_end - decay_start)
        progress = min(max(epoch - decay_start, 0.0), total)
        factor = floor + (1 - floor) * 0.5 * (1 + torch.cos(torch.tensor(progress / total * torch.pi)).item())
    return float(base_weight * factor)


def adaptive_global_chunk_spans(
    caption: str,
    tokenizer,
    *,
    base_window: int = 3,
    stride: int = 1,
    max_spans: int = 64,
):
    if not isinstance(caption, str) or not caption.strip():
        return []

    encoding = tokenizer(caption, return_offsets_mapping=True, add_special_tokens=True)
    words = list(re.finditer(r"\b\w+\b", caption))
    if not words:
        return []

    sentence_bounds = []
    begin = 0
    for punct in re.finditer(r"[.,;!?]+", caption):
        sentence_bounds.append((begin, punct.end()))
        begin = punct.end()
    if begin < len(caption):
        sentence_bounds.append((begin, len(caption)))

    spans = []
    word_ptr = 0
    for sent_start, sent_end in sentence_bounds:
        start_word_ptr = word_ptr
        while word_ptr < len(words) and words[word_ptr].start() < sent_end:
            word_ptr += 1
        end_word_ptr = word_ptr - 1
        sent_word_len = end_word_ptr - start_word_ptr + 1
        if sent_word_len <= 0:
            continue

        if sent_word_len <= 4:
            window = 2
        elif sent_word_len <= 10:
            window = base_window
        elif sent_word_len <= 20:
            window = base_window + 1
        elif sent_word_len <= 40:
            window = base_window + 2
        else:
            window = base_window + 3

        for i in range(start_word_ptr, end_word_ptr - window + 2, stride):
            start_char = words[i].start()
            end_char = words[i + window - 1].end()
            token_start = encoding.char_to_token(start_char)
            token_end = encoding.char_to_token(end_char - 1)
            if token_start is not None and token_end is not None and token_end >= token_start:
                spans.append((token_start, token_end + 1))
            if len(spans) >= max_spans:
                break

        if len(spans) >= max_spans:
            break

    return spans


def compute_structure_centric_loss(
    org_scribble_embeds: torch.Tensor,
    filtered_org_text_embeds: torch.Tensor,
    has_org_scribble: torch.Tensor,
    *,
    logit_scale: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    has_structure = has_org_scribble.bool()
    if not has_structure.any():
        return torch.tensor(0.0, device=org_scribble_embeds.device)

    image_feats = F.normalize(org_scribble_embeds[has_structure] + eps, dim=-1)
    text_feats = F.normalize(filtered_org_text_embeds[has_structure] + eps, dim=-1)
    return clip_loss(logit_scale * (image_feats @ text_feats.t()))


def compute_rgb_structure_consistency_loss(
    org_image_embeds: torch.Tensor,
    org_scribble_embeds: torch.Tensor,
    has_org_scribble: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    has_structure = has_org_scribble.bool()
    if not has_structure.any():
        return torch.tensor(0.0, device=org_image_embeds.device)

    return F.cosine_embedding_loss(
        F.normalize(org_image_embeds[has_structure] + eps, dim=-1),
        F.normalize(org_scribble_embeds[has_structure] + eps, dim=-1),
        torch.ones(has_structure.sum(), device=org_image_embeds.device),
    )


def compute_local_structure_centric_loss(
    model,
    org_text_tokens: torch.Tensor,
    filtered_texts: Sequence[str],
    edge_seg_embeds_flat: torch.Tensor,
    edge_valid_mask: torch.Tensor,
    tokenizer,
    *,
    chunk_base_window: int,
    chunk_stride: int,
    chunk_tau: float,
    eps: float,
):
    batch_size, top_k = edge_valid_mask.shape

    edge_all = F.normalize(edge_seg_embeds_flat + eps, dim=-1)
    valid_all_mask = edge_valid_mask.reshape(-1).to(edge_all.device)

    neg_inf = torch.finfo(edge_all.dtype).min
    tau = float(max(1e-6, chunk_tau))

    debug_num_chunks = []
    debug_num_pos = []
    debug_pos_ratio = []
    debug_margin = []
    debug_entropy = []

    with torch.no_grad():
        effective_scale = float((model.logit_scale.exp() / tau).item())

    total_loss = torch.tensor(0.0, device=edge_all.device)
    valid_items = 0

    for b in range(batch_size):
        chunk_spans = adaptive_global_chunk_spans(
            filtered_texts[b],
            tokenizer,
            base_window=chunk_base_window,
            stride=chunk_stride,
            max_spans=64,
        )
        if not chunk_spans:
            continue

        text_tokens_b = org_text_tokens[b]
        chunk_embeddings = []
        for start, end in chunk_spans:
            if 0 <= start < end <= text_tokens_b.shape[0]:
                with torch.no_grad():
                    weights = F.softmax(text_tokens_b[start:end].norm(dim=-1), dim=0)
                chunk_embeddings.append((text_tokens_b[start:end] * weights[:, None]).sum(dim=0))

        if not chunk_embeddings:
            continue

        chunk_embeds = torch.stack(chunk_embeddings, dim=0)
        chunk_embeds = model.text_model.final_layer_norm(chunk_embeds)
        chunk_embeds = model.text_projection(chunk_embeds)
        chunk_embeds_n = F.normalize(chunk_embeds + eps, dim=-1)
        num_chunks = chunk_embeds_n.shape[0]

        scale_eff = (model.logit_scale.exp() / tau).clamp(max=100.0)
        sim_all = scale_eff * (chunk_embeds_n @ edge_all.t())

        sim_den = sim_all.masked_fill(~valid_all_mask.bool().unsqueeze(0), neg_inf)
        log_den = torch.logsumexp(sim_den, dim=1)

        pos_mask = torch.zeros(batch_size * top_k, dtype=torch.bool, device=edge_all.device)
        sample_pos_mask = edge_valid_mask[b].to(edge_all.device)
        if not sample_pos_mask.any():
            continue

        offset = b * top_k
        pos_indices = torch.arange(offset, offset + top_k, device=edge_all.device)[sample_pos_mask]
        pos_mask[pos_indices] = True

        sim_pos = sim_all.masked_fill(~pos_mask.unsqueeze(0), neg_inf)
        log_pos = torch.logsumexp(sim_pos, dim=1)

        total_loss += -(log_pos - log_den).mean()
        valid_items += 1

        with torch.no_grad():
            debug_num_chunks.append(float(num_chunks))
            debug_num_pos.append(float(sample_pos_mask.sum().item()))
            debug_pos_ratio.append(float(torch.exp(log_pos - log_den).mean().item()))

            neg_mask = valid_all_mask & (~pos_mask)
            if neg_mask.any():
                top_pos = sim_all[:, pos_mask].max(dim=1).values
                top_neg = sim_all[:, neg_mask].max(dim=1).values
                debug_margin.append(float((top_pos - top_neg).mean().item()))

            log_probs = sim_den - log_den.unsqueeze(1)
            probs = torch.exp(log_probs)
            entropy = -(probs * log_probs).sum(dim=1).mean().item()
            debug_entropy.append(float(entropy))

    if valid_items > 0:
        total_loss = total_loss / valid_items
    else:
        total_loss = torch.tensor(0.0, device=edge_all.device)

    debug = {
        "effective_scale": effective_scale,
        "avg_chunks": float(sum(debug_num_chunks) / len(debug_num_chunks)) if debug_num_chunks else 0.0,
        "avg_pos": float(sum(debug_num_pos) / len(debug_num_pos)) if debug_num_pos else 0.0,
        "pos_mass_ratio": float(sum(debug_pos_ratio) / len(debug_pos_ratio)) if debug_pos_ratio else 0.0,
        "top1_margin": float(sum(debug_margin) / len(debug_margin)) if debug_margin else 0.0,
        "entropy": float(sum(debug_entropy) / len(debug_entropy)) if debug_entropy else 0.0,
        "valid_items": int(valid_items),
    }
    return total_loss, debug
