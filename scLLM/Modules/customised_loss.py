import torch
import torch.nn.functional as F


def masked_mse_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the masked MSE loss between input and target.
    """
    mask = mask.float()
    loss = F.mse_loss(input * mask, target * mask, reduction="sum")
    return loss / mask.sum()


def criterion_neg_log_bernoulli(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the negative log-likelihood of Bernoulli distribution
    """
    mask = mask.float()
    bernoulli = torch.distributions.Bernoulli(probs=input)
    masked_log_probs = bernoulli.log_prob((target > 0).float()) * mask
    return -masked_log_probs.sum() / mask.sum()


def masked_relative_error(
    input: torch.Tensor, target: torch.Tensor, mask: torch.LongTensor
) -> torch.Tensor:
    """
    Compute the masked relative error between input and target.
    """
    assert mask.any()
    loss = torch.abs(input[mask] - target[mask]) / (target[mask] + 1e-6)
    return loss.mean()


import torch


def _ranknet_loss(logits_i, logits_j, label_i, label_j):
    # compute_probability(logits_i, logits_j)
    P_ij= torch.sigmoid(logits_i - logits_j)
    # compute loss
    sig1 = (label_i.detach() > label_j.detach()).float()
    sig2 = (label_i.detach() < label_j.detach()).float()
    loss = - sig1 * torch.log(P_ij) - sig2 * torch.log(1 - P_ij)
    return loss

import torch
import torch.nn as nn

class RankNetLoss(nn.Module):
    def __init__(self):
        super(RankNetLoss, self).__init__()

    def forward(self, logits_i, logits_j, labels_i, labels_j):
        loss = _ranknet_loss(logits_i, logits_j, labels_i, labels_j)
        return loss