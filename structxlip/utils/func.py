import torch
import torch.nn as nn


def longclip_pos_embeddings(model, new_max_token: int, keep_len: int = 20):
    """
    Extend CLIP text positional embeddings to a longer token length.

    Strategy:
    1. Keep the first `keep_len` positions unchanged.
    2. For remaining positions, perform 4-step interpolation between adjacent positions.
    3. Extrapolate a short tail from the final two original embeddings.
    """
    text_model = model.text_model
    pos_embeddings_pre = text_model.embeddings.position_embedding.weight
    length, dim = pos_embeddings_pre.shape

    max_len = 8 * length - 7 * keep_len
    if new_max_token > max_len:
        raise ValueError(
            f"new_max_token ({new_max_token}) > max supported ({max_len}) "
            f"for orig_len={length}, keep_len={keep_len}"
        )

    pos_new = torch.zeros(
        [new_max_token, dim],
        dtype=pos_embeddings_pre.dtype,
        device=pos_embeddings_pre.device,
    )

    upto = min(keep_len, new_max_token)
    if upto > 0:
        pos_new[:upto] = pos_embeddings_pre[:upto]

    write_ptr = keep_len
    steps = max(0, length - 1 - keep_len)
    for i in range(steps):
        a = pos_embeddings_pre[keep_len + i]
        b = pos_embeddings_pre[keep_len + i + 1]
        slots = [
            a,
            (3 * a + 1 * b) / 4,
            (2 * a + 2 * b) / 4,
            (1 * a + 3 * b) / 4,
        ]
        for s in slots:
            if write_ptr < new_max_token:
                pos_new[write_ptr] = s
                write_ptr += 1

    delta_extrap = pos_embeddings_pre[-1] - pos_embeddings_pre[-2]
    tail = [
        pos_embeddings_pre[-1] + 0 * delta_extrap / 4,
        pos_embeddings_pre[-1] + 1 * delta_extrap / 4,
        pos_embeddings_pre[-1] + 2 * delta_extrap / 4,
        pos_embeddings_pre[-1] + 3 * delta_extrap / 4,
    ]
    for s in tail:
        if write_ptr < new_max_token:
            pos_new[write_ptr] = s
            write_ptr += 1

    text_model.embeddings.position_embedding.weight = nn.Parameter(pos_new)
    new_position_ids = torch.arange(0, new_max_token, device=pos_new.device).unsqueeze(0)
    if hasattr(text_model.embeddings, "position_ids"):
        text_model.embeddings.position_ids = new_position_ids
    else:
        text_model.embeddings.register_buffer("position_ids", new_position_ids)


def batch_align(fabric, x: torch.Tensor, grads: bool = True) -> torch.Tensor:
    """Gather tensor features across processes and flatten to global batch shape."""
    if fabric is None or not hasattr(fabric, "all_gather"):
        return x
    if not grads:
        return x

    try:
        y = fabric.all_gather(x, sync_grads=True)
    except TypeError:
        y = fabric.all_gather(x)

    y = y.contiguous()
    if y.dim() >= 3:
        y = y.view(y.shape[0] * y.shape[1], *y.shape[2:])
    return y


def print_trainable_parameters(fabric, model: nn.Module):
    trainable_params = 0
    all_params = 0

    for _, param in model.named_parameters():
        n = param.numel()
        all_params += n
        if param.requires_grad:
            trainable_params += n

    ratio = 100.0 * trainable_params / max(1, all_params)
    msg = (
        f"trainable params: {trainable_params} || "
        f"all params: {all_params} || trainable%: {ratio:.2f}"
    )

    if hasattr(fabric, "print"):
        fabric.print(msg)
        try:
            device = fabric.device if hasattr(fabric, "device") else next(model.parameters()).device
            fabric.print(
                f"Current CUDA memory allocated: "
                f"{torch.cuda.memory_allocated(device=device)} bytes"
            )
        except Exception:
            pass
    else:
        print(msg)
