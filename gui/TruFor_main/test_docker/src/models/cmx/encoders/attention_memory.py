"""Inference-only query chunking for the original global attention formula.

Every query still attends to every key. This does not resize or tile the image,
change weights, restrict context, or approximate softmax. It bounds the score
matrix allocation; the model's other activations still need memory.
"""
import torch

SCORE_BUDGET = 512 * 1024 * 1024


def bounded_attention(q, k, v, scale, dropout, budget=SCORE_BUDGET):
    row_bytes = q.shape[0] * q.shape[1] * k.shape[-2] * q.element_size()
    chunk = max(1, budget // row_bytes)
    if torch.is_grad_enabled() or q.shape[-2] <= chunk:
        scores = (q @ k.transpose(-2, -1)) * scale
        return dropout(scores.softmax(dim=-1)) @ v
    output = torch.empty_like(q)
    keys = k.transpose(-2, -1)
    for start in range(0, q.shape[-2], chunk):
        stop = min(start + chunk, q.shape[-2])
        scores = q[..., start:stop, :] @ keys
        scores.mul_(scale)
        probabilities = dropout(scores.softmax(dim=-1))
        del scores
        output[..., start:stop, :] = probabilities @ v
        del probabilities
    return output
