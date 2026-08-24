"""Pairwise Uncertainty-guided Curriculum Learning (PUCL).

The rank-modulated uncertainty schedule is adapted from the MIT-licensed
SpatialCL implementation, with a self-contained two-view public interface.
"""

import math

import torch
import torch.nn.functional as F


def _validate_inputs(
    embeddings, labels, epoch, total_epochs, temperature, eps
):
    if not torch.is_tensor(embeddings) or embeddings.ndim != 3:
        raise ValueError("embeddings must have rank 3: [batch, views, features]")
    if embeddings.shape[1] != 2:
        raise ValueError("views must contain exactly 2 embeddings per sample")
    if embeddings.shape[2] < 1:
        raise ValueError("embeddings feature dimension must be positive")

    labels = torch.as_tensor(labels, device=embeddings.device)
    if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
        raise ValueError("labels must have one entry per embedding sample")

    if isinstance(total_epochs, bool) or not isinstance(total_epochs, (int, float)):
        raise ValueError("total_epochs must be positive")
    if not math.isfinite(float(total_epochs)) or total_epochs <= 0:
        raise ValueError("total_epochs must be positive")
    if isinstance(epoch, bool) or not isinstance(epoch, (int, float)):
        raise ValueError("epoch must be between 0 and total_epochs")
    if not math.isfinite(float(epoch)) or epoch < 0 or epoch > total_epochs:
        raise ValueError("epoch must be between 0 and total_epochs")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("temperature must be positive")
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be positive")
    if isinstance(eps, bool) or not isinstance(eps, (int, float)):
        raise ValueError("eps must be positive")
    if not math.isfinite(float(eps)) or eps <= 0:
        raise ValueError("eps must be positive")
    return labels.to(device=embeddings.device, dtype=torch.long)


def pucl_loss(
    embeddings,
    labels,
    *,
    epoch,
    total_epochs,
    temperature=0.1,
    eps=1e-12,
):
    """Return the scalar PUCL objective for two views of each sample."""
    labels = _validate_inputs(
        embeddings, labels, epoch, total_epochs, temperature, eps
    )
    batch_size, views, feature_size = embeddings.shape
    work_embeddings = (
        embeddings.float()
        if embeddings.dtype in (torch.float16, torch.bfloat16)
        else embeddings
    )
    if batch_size == 0:
        return work_embeddings.sum() * 0.0

    work_eps = max(float(eps), torch.finfo(work_embeddings.dtype).tiny)
    features = F.normalize(
        work_embeddings.reshape(batch_size * views, feature_size),
        dim=1,
        eps=1e-8,
    )
    flat_labels = labels.repeat_interleave(views)
    image_ids = torch.arange(batch_size, device=embeddings.device).repeat_interleave(
        views
    )

    # Subjective-logic co-cluster uncertainty from the MIT-licensed source.
    uncertainty_similarity = features @ features.transpose(0, 1)
    uncertainty_similarity.fill_diagonal_(0.0)
    positive_evidence = torch.exp(F.softmax(uncertainty_similarity, dim=1))
    negative_evidence = torch.exp(F.softmax(1.0 - uncertainty_similarity, dim=1))
    uncertainty = 2.0 / (positive_evidence + negative_evidence + 2.0)
    uncertainty.masked_fill_(
        ~flat_labels[:, None].eq(flat_labels[None, :]), 0.0
    )

    pair_count = features.shape[0]
    self_mask = torch.eye(pair_count, dtype=torch.bool, device=embeddings.device)
    same_image = image_ids[:, None].eq(image_ids[None, :])
    same_label = flat_labels[:, None].eq(flat_labels[None, :])
    same_label = same_label.clone()
    same_label[self_mask] = False
    strong_positive = same_label & same_image
    weak_positive = same_label & ~same_image
    positive_mask = strong_positive | weak_positive
    negative_mask = ~same_label

    descending = torch.argsort(uncertainty, dim=1, descending=True)
    ranks = torch.zeros_like(uncertainty, dtype=torch.float32)
    rank_values = torch.arange(
        pair_count, dtype=torch.float32, device=embeddings.device
    ).expand(pair_count, -1)
    ranks.scatter_(1, descending, rank_values)
    rank_fraction = ranks / float(pair_count)
    progress = float(epoch) / float(total_epochs)
    curriculum_weight = 1.0 + torch.exp(-progress * rank_fraction)

    positive_weights = strong_positive.float()
    positive_weights = positive_weights + curriculum_weight * weak_positive.float()
    positive_weights.masked_fill_(~positive_mask, 0.0)

    similarities = features @ features.transpose(0, 1)
    similarities = similarities / float(temperature)
    similarities = similarities - similarities.max(dim=1, keepdim=True)[0].detach()
    exp_similarities = torch.exp(similarities).clamp_min(work_eps)
    numerator = exp_similarities * positive_weights

    exp_negatives = exp_similarities * negative_mask.float()
    negative_sum = exp_negatives.sum(dim=1, keepdim=True).clamp_min(work_eps)
    negative_weights = exp_negatives / negative_sum
    negative_term = negative_weights * exp_similarities * negative_mask.float()

    denominator = (numerator + negative_term.sum(dim=1, keepdim=True)).clamp_min(
        work_eps
    )
    log_probability = torch.log((numerator / denominator).clamp_min(work_eps))
    loss_matrix = -log_probability * positive_mask.float()

    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not torch.any(valid):
        return work_embeddings.sum() * 0.0
    anchor_losses = loss_matrix.sum(dim=1)[valid] / (
        positive_count[valid] + work_eps
    )
    return anchor_losses.mean()


__all__ = ["pucl_loss"]
