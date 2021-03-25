"""

Script to test lesion correspondences based on the algorithm in Cheblus et al. (2018)

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import numpy as np
import SimpleITK as sitk
import os
from segmentation_metrics.metrics import *
from segmentation_metrics.lesion_correspondence import *

PAT_DIR = '/home/ishaan/lesion_segmentation/checkpoints/baseline/unet_full_slice_deeper/images_499/72'

if __name__ == '__main__':

    seg_itk = sitk.ReadImage(os.path.join(PAT_DIR, 'binary_post_proc_pred.nii.gz'))
    gt_itk = sitk.ReadImage(os.path.join(PAT_DIR, 'true_mask.nii.gz'))

    seg_np = sitk.GetArrayFromImage(seg_itk).transpose((1, 2, 0))
    gt_np = sitk.GetArrayFromImage(gt_itk).transpose((1, 2, 0))

    dgraph = create_correspondence_graph(seg=seg_np, gt=gt_np)
    count_detections(dgraph)
