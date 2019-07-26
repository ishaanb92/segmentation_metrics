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
    assert (isinstance(seg, np.ndarray))
    assert (isinstance(gt, np.ndarray))

    eps = 0.0001
    seg = seg.flatten()
    gt = gt.flatten()

    inter = np.dot(seg, gt)
    union = np.sum(seg) + np.sum(gt)

    dice_similarity_coeff = (2*inter)/(union + eps)

    return dice_similarity_coeff


def hausdorff_distance(seg, gt):
    """
    Calculate the symmetric hausdorff distance between
    the segmentation and ground truth
    This metric is also known as Maximum Surface Distance

    :param seg:
    :param gt:
    :return:
    """
    assert (isinstance(seg, np.ndarray))
    assert (isinstance(gt, np.ndarray))

    msd = max(directed_hausdorff_distance(seg, gt),
              directed_hausdorff_distance(gt, seg))

    return msd


def directed_hausdorff_distance(vol1, vol2):
    """
    Directed Hausdorff distance between a pair of (3+1)-D volumes
    Max over hausdorff distances calculated between aligned slice pairs
    FIXME: Extend to N-D
    FIXME: Check for logical bugs
    :param vol1: Expected dimensions num_slices x num_classes x H x W
    :param vol2: Expected dimensions num_slices x num_classes x H x W
    :return:
    """
    assert (isinstance(vol1, np.ndarray) and isinstance(vol2, np.ndarray))
    assert(vol1.ndim == 4 and vol2.ndim == 4)

    # We only need the foreground class
    vol1 = vol1[:, 1, :, :]
    vol2 = vol2[:, 1, :, :]

    n_slices = vol1.shape[0]

    hausdorff_distance_slice_pair = []
    for slice_id in range(n_slices):
        hausdorff_distance_slice_pair.append(directed_hausdorff(vol1[slice_id, :, :], vol2[slice_id, :, :])[0])

    directed_hd = max(hausdorff_distance_slice_pair)

    return directed_hd


def relative_volume_difference(seg, gt):
    """
    Calculate the relative volume difference between segmentation
    and the ground truth
    :param seg:
    :param gt:
    :return:
    """

    assert (isinstance(seg, np.ndarray))
    assert (isinstance(gt, np.ndarray))

    rvd = (np.sum(gt, axis=None) - np.sum(seg, axis=None))/np.sum(gt, axis=None)
    return rvd
