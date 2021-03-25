"""
Script to establish lesion correspondence, since there isn't necessarily 1:1 corr. between objects in the
predicted and reference segmentations

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import os
from .metrics import dice_score
from utils.image_utils import return_lesion_coordinates
import networkx as nx
import numpy as np
from networkx.algorithms import bipartite


class Lesion():

    def __init__(self, coordinates:slice, idx:int=-1, predicted:bool=True):

        self.coordinates = coordinates

        if predicted is True:
            self.name = 'Predicted_lesion_{}'.format(idx)
        else:
            self.name = 'Reference_lesion_{}'.format(idx)

        self.label = -1

    def get_coordinates(self):
        return self.coordinates

    def get_name(self):
        return self.name

    # Use this method only for predicted lesion!!
    def set_label(self, label):
        self.label = label

    def get_label(self):
        return self.label


def create_correspondence_graph(seg, gt):

    assert(isinstance(seg, np.ndarray))
    assert(isinstance(gt, np.ndarray))

    # Find connected components
    predicted_slices, num_predicted_lesions = return_lesion_coordinates(mask=seg)
    true_slices, num_true_lesions = return_lesion_coordinates(mask=gt)
    print('Number of predicted lesion = {}'.format(num_predicted_lesions))
    print('Number of true lesions ={}'.format(num_true_lesions))

    pred_lesions = []
    gt_lesions = []

    for idx, pred_slice in enumerate(predicted_slices):
        pred_lesions.append(Lesion(coordinates=pred_slice,
                                   idx=idx,
                                   predicted=True))

    for idx, gt_slice in enumerate(true_slices):
        gt_lesions.append(Lesion(coordinates=gt_slice,
                                 idx=idx,
                                 predicted=False))


    # Create a directed bipartite graph
    dgraph = nx.DiGraph()

    # In case of no overlap between 2 lesion, we add an edge with weight 0
    # so that the graph is a valid bipartite graph

    # Create forward edges (partition 0 -> partition 1)
    for pred_lesion in pred_lesions:
        seg_lesion_volume = np.zeros_like(seg)
        lesion_slice = pred_lesion.get_coordinates()
        seg_lesion_volume[lesion_slice] += seg[lesion_slice]
        # Iterate over GT lesions
        for gt_lesion in gt_lesions:
            gt_lesion_volume = np.zeros_like(gt)
            gt_lesion_slice = gt_lesion.get_coordinates()
            gt_lesion_volume[gt_lesion_slice] += gt[gt_lesion_slice]
            # Compute overlap
            dice = dice_score(seg_lesion_volume, gt_lesion_volume)
            if dice > 0:
                dgraph.add_weighted_edges_from([(pred_lesion, gt_lesion, dice)])
            else:
                dgraph.add_weighted_edges_from([(pred_lesion, gt_lesion, 0)]) # False positive


    # Create backward edges (partition 1 -> partition 0)
    for gt_lesion in gt_lesions:
        gt_lesion_volume = np.zeros_like(gt)
        gt_lesion_slice = gt_lesion.get_coordinates()
        gt_lesion_volume[gt_lesion_slice] += seg[gt_lesion_slice]
        # Iterate over GT lesions
        for pred_lesion in pred_lesions:
            seg_lesion_volume = np.zeros_like(seg)
            lesion_slice = pred_lesion.get_coordinates()
            seg_lesion_volume[lesion_slice] += seg[lesion_slice]
            # Compute overlap
            dice = dice_score(seg_lesion_volume, gt_lesion_volume)
            if dice > 0:
                dgraph.add_weighted_edges_from([(gt_lesion, pred_lesion, dice)])
            else:
                dgraph.add_weighted_edges_from([(gt_lesion, pred_lesion, 0)])

    # Check if the constructed graph is bipartite
    assert(bipartite.is_bipartite(dgraph))

    return dgraph

def count_detections(dgraph=None):

    print('Directed graph has {} nodes'.format(dgraph.number_of_nodes()))

    pred_lesion_nodes, gt_lesion_nodes = bipartite.sets(dgraph)

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Count true positives and false negatives
    for gt_lesion_node in gt_lesion_nodes:
        incoming_edge_weights = []

        for pred_lesion_node in pred_lesion_nodes:
            # Examine edge weights
            edge_weight = dgraph[pred_lesion_node][gt_lesion_node]['weight']
            incoming_edge_weights.append(edge_weight)

        # Check the maximum weight
        max_weight = np.amax(np.array(incoming_edge_weights))
        if max_weight > 0: # Atleast one incoming edge with dice > 0
            true_positives += 1
        else:
            false_negatives += 1

    # Count false positives
    for pred_lesion_node in pred_lesion_nodes:
        outgoing_edge_weights = []

        for gt_lesion_node in gt_lesion_nodes:
            edge_weight = dgraph[pred_lesion_node][gt_lesion_node]['weight']
            outgoing_edge_weights.append(edge_weight)

        # Check maximum weight
        max_weight = np.amax(np.array(outgoing_edge_weights))
        if max_weight == 0:
            false_positives += 1
            pred_lesion_node.set_label(label=1)
        else:
            pred_lesion_node.set_label(label=0)


    print('Number of detected lesions = {}'.format(true_positives))
    print('Number of false positives = {}'.format(false_positives))
    print('Number of false negatives = {}'.format(false_negatives))




