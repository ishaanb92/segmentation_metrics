import numpy as np
from scipy.spatial.distance import directed_hausdorff
from utils.image_utils import find_number_of_objects

eps = 0.0001  # Avoid div-by-zero situations


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

    inter = calculate_intersection(seg, gt)
    union = np.sum(seg) + np.sum(gt)

    dice_similarity_coeff = (2*inter)/(union + eps)

    return dice_similarity_coeff


def calculate_intersection(seg, gt):
    """
    Calculates intersection (as dot product) between 2 masks

    :param seg:
    :param gt:
    :return:
    """
    assert (isinstance(seg, np.ndarray))
    assert (isinstance(gt, np.ndarray))

    if seg.ndim > 1:
        seg = seg.flatten()
    if gt.ndim > 1:
        gt = gt.flatten()

    return np.dot(seg, gt)


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
    assert(vol1.ndim == 3 and vol2.ndim == 3)  # HD for 3D volumes

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

    RVD (A, B) = (|B| - |A|)/|B|

    If RVD > 0 => Under-segmentation
       RVD < 0 => Over-segmentation

    :param seg: (numpy ndarray) Predicted segmentation
    :param gt: (numpy ndarray) Ground truth mask
    :return: rvd: (float) Relative volume difference (as %)
    """

    assert (isinstance(seg, np.ndarray))
    assert (isinstance(gt, np.ndarray))

    rvd = (np.sum(gt, axis=None) - np.sum(seg, axis=None))/(np.sum(gt, axis=None) + eps)
    rvd = rvd*100

    return rvd


def analyze_detected_lesions(seg, gt, verbose=False):
    """
    Function to analyze the detected lesions i.e. count the number of true positives, false negatives and
    false positives.
    Presence is determined on the basis of a non-zero overlap i.e. if the intersection (dot product) > 0,

    :param seg: (numpy ndarray)
    :param gt: (numpy ndarray)
    :return: lesion_counts: (dict) Dictionary with counts of different types of lesions
    """

    assert (isinstance(seg, np.ndarray))
    assert (isinstance(gt, np.ndarray))

    num_predicted_lesions, list_of_preds = find_number_of_objects(mask=seg)
    num_true_lesions, list_of_lesions = find_number_of_objects(mask=gt)

    true_positives = 0

    for true_volumes in list_of_lesions:
        intersection = calculate_intersection(seg[true_volumes], gt[true_volumes])
        if intersection > 0:  # This lesion is considered detected
            true_positives += 1

    false_negatives = num_true_lesions-true_positives
    false_positives = num_predicted_lesions-true_positives

    lesion_counts = {'true positives': true_positives,
                     'false negatives': false_negatives,
                     'false positives': false_positives,
                     'true lesions': num_true_lesions}

    if verbose is True:
        print('Number of lesions in GT = {}\n'
              'Number of lesions in prediction = {}\n'
              'True positives detected = {}\n'
              'False negatives (missed lesions) = {}\n'
              'False positives detected = {}'.format(num_true_lesions,
                                                     num_predicted_lesions,
                                                     true_positives,
                                                     false_negatives,
                                                     false_positives))

    return lesion_counts


def calculate_true_positive_rate(seg, gt):
    """
    Calculate the TPR between prediction and ground truth

    :param seg: (numpy ndarray) N_CLASSES x H x W x D
    :param gt: (numpy ndarray) N_CLASSES x H x W x D
    :return: tpr: (float) True positive rate
    """

    assert (isinstance(seg, np.ndarray))
    assert (isinstance(gt, np.ndarray))

    lesion_counts = analyze_detected_lesions(seg, gt, verbose=True)
    tpr = lesion_counts['true positives']/(lesion_counts['true positives'] + lesion_counts['false negatives'] + eps)

    return tpr

