import torch
import numpy as np


def dice_score(seg, gt):
    """
    Function that calculates the dice similarity co-efficient
    over the entire batch

    Parameters:
        seg (torch.Tensor) : Batch of (Predicted )Segmentation map
        gt (torch.Tensor) : Batch of ground truth maps

    Returns:
        dice_similarity_coeff (float) : Dice similiarty between predicted segmentations and ground truths

    """
    with torch.no_grad():
        if isinstance(seg, np.ndarray):
            seg = torch.Tensor(seg)
        if isinstance(gt, np.ndarray):
            gt = torch.Tensor(gt)

        seg = seg.contiguous().view(-1)
        gt = gt.contiguous().view(-1)

        inter = torch.dot(seg, gt).item()
        union = torch.sum(seg) + torch.sum(gt)

        eps = 0.0001  # For numerical stability

        dice_similarity_coeff = (2*inter)/(union.item() + eps)

        return dice_similarity_coeff
