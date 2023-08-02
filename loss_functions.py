"""
Modified pytorch's smoothl1loss function to include CQL loss function.
"""
import torch
import torch.nn.functional as F


from torch import Tensor



def CqlLoss(input: Tensor, target:Tensor, lam, reduction="sum") -> Tensor:
    """
    Modified loss for CQL
    """
    # change smooth_l1_loss to mse_loss

    return (F.mse_loss(input, target, reduction=reduction) + lam * input).mean()

