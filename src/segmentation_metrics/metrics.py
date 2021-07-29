import numpy as np
from scipy.spatial.distance import directed_hausdorff
from utils.image_utils import return_lesion_coordinates
from scipy.ndimage import binary_opening, binary_closing, generate_binary_structure, binary_dilation
eps = 1e-8  # Avoid div-by-zero situations


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

    return np.sum(np.multiply(seg, gt))


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


def compute_spatial_entropy(seg, gt, umap):
    """
    Function to (qualitatively) analyze uncertainty map
    We compute average entropy over regions where a lesion has been predicted
    to check if the avg. entropy is higher for false postives.

    """
    assert(isinstance(seg, np.ndarray))
    assert(isinstance(gt, np.ndarray))
    assert(isinstance(umap, np.ndarray))

    predicted_slices, num_predicted_lesions = return_lesion_coordinates(mask=seg)
    true_slices, num_true_lesions = return_lesion_coordinates(mask=gt)
    # Analysis for true postives and false positives
    region_uncertainties_tp = []
    region_uncertainties_fp = []
    region_uncertainties_fn = []

    for predicted_volume in predicted_slices:
        intersection = dice_score(seg[predicted_volume], gt[predicted_volume])
        if intersection > 0: # True positive
            region_uncertainties_tp.append(np.mean(umap[predicted_volume]))
        else: # False Positive
            region_uncertainties_fp.append(np.mean(umap[predicted_volume]))

   # Analysis for false negatives
    for true_volume in true_slices:
        intersection = dice_score(seg[true_volume], gt[true_volume])
        if intersection == 0: # False negative
            region_uncertainties_fn.append(np.mean(umap[true_volume]))

    region_unc_dict = {'tp_unc' : region_uncertainties_tp, 'fp_unc': region_uncertainties_fp, 'fn_unc': region_uncertainties_fn}

    return region_unc_dict

def compute_lesion_volumes(seg, gt):
    """
    Function to compute (approximate) lesion volume.
    The find_objects() function provides a tuple of slices defining
    the minimal parallelopiped covering the lesion
    The volume is given as length*breadth*depth

    """
    predicted_slices, num_predicted_lesions = return_lesion_coordinates(mask=seg)
    true_slices, num_true_lesions = return_lesion_coordinates(mask=gt)

    tp_pred_volumes = []
    fp_pred_volumes = []
    fn_true_volumes = []
    tp_true_volumes = []

    for predicted_volume in predicted_slices:
        intersection = dice_score(seg[predicted_volume], gt[predicted_volume])
        length, breadth, depth = seg[predicted_volume].shape
        volume = length*breadth*depth
        if intersection > 0: # True positive
            tp_pred_volumes.append(volume)
        else: # False Positive
            fp_pred_volumes.append(volume)

    for true_volume in true_slices:
        intersection = dice_score(seg[true_volume], gt[true_volume])
        length, breadth, depth = gt[true_volume].shape
        volume = length*breadth*depth
        if intersection == 0: # False negative
            fn_true_volumes.append(volume)
        else:
            tp_true_volumes.append(volume)

    lesion_volume_dict = {'tp_pred': tp_pred_volumes,
                          'tp_true': tp_true_volumes,
                          'fp_pred': fp_pred_volumes,
                          'fn_true': fn_true_volumes}

    return lesion_volume_dict

