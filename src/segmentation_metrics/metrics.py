import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff


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


def symmetric_hausdorff_distance(seg, gt):
    """
    Calculate the symmetric hausdorff distance between
    the segmentation and ground truth
    This metric is also known as Maximum Surface Distance

    :param seg:
    :param gt:
    :return:
    """
    if isinstance(seg, torch.Tensor):
        seg = seg.numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.numpy()

    msd = max(directed_hausdorff(seg[:, 1, :, :], gt[:, 1, :, :])[0],
              directed_hausdorff(gt[:, 1, :, :], seg[:, 1, :, :])[0])

    return msd


def relative_volume_difference(seg, gt):
    """
    Calculate the relative volume difference between segmentation
    and the ground truth
    :param seg:
    :param gt:
    :return:
    """

    if isinstance(seg, torch.Tensor):
        seg = seg.numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.numpy()

    rvd = (np.sum(gt, axis=None) - np.sum(seg, axis=None))/np.sum(gt, axis=None)
    return rvd
