import numpy as np
from scipy.spatial.distance import directed_hausdorff

eps = 0.0001  # Avoid 0/0 situations


def dice_score(seg, gt):
    """
    Function that calculates the dice similarity co-efficient
    over the entire batch

    :param seg: (numpy ndarray) Batch of (Predicted )Segmentation map
    :param gt: (numpy ndarray) Batch of ground truth maps

    :return dice_similarity_coeff: (float) Dice score

    """
    assert (isinstance(seg, np.ndarray))
    assert (isinstance(gt, np.ndarray))

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

    :param seg: (numpy ndarray) Predicted segmentation. Expected dimensions num_slices x H x W
    :param gt: (numpy ndarray) Ground Truth. Expected dimensions num_slices x H x W
    :return: msd: (numpy ndarray) Symmetric hausdorff distance (Maximum surface distance)
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
    FIXME: Currently works for a 2-class label (foreground + background)
    FIXME: Check for logical bugs

    :param vol1: (numpy ndarray) Expected dimensions num_slices x H x W
    :param vol2: (numpy ndarray) Expected dimensions num_slices x H x W
    :return: directed_hd : (float) Directed Hausdorff distance
    """
    assert (isinstance(vol1, np.ndarray) and isinstance(vol2, np.ndarray))
    assert(vol1.ndim == 3 and vol2.ndim == 3) # HD for 3D volumes

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

    RVD (A, B) = (|B| - |A|)/|A|

    If RVD > 0 => Under-segmentation
       RVD < 0 => Over-segmentation

    :param seg: (numpy ndarray) Predicted segmentation
    :param gt: (numpy ndarray) Ground truth mask
    :return: rvd: (float) Relative volume difference (as %)
    """

    assert (isinstance(seg, np.ndarray))
    assert (isinstance(gt, np.ndarray))

    rvd = (np.sum(gt, axis=None) - np.sum(seg, axis=None))/(np.sum(seg, axis=None) + eps)
    rvd = rvd*100

    return rvd
