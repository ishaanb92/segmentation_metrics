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
import matplotlib.pyplot as plt

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


def create_correspondence_graph(seg, gt, verbose=False):

    assert(isinstance(seg, np.ndarray))
    assert(isinstance(gt, np.ndarray))

    # Find connected components
    predicted_slices, num_predicted_lesions = return_lesion_coordinates(mask=seg)
    true_slices, num_true_lesions = return_lesion_coordinates(mask=gt)

    if verbose is True:
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
    # so that the graph has no disconnected nodes

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
            else: # False positive
                dgraph.add_weighted_edges_from([(pred_lesion, gt_lesion, 0)])


    # Create backward edges (partition 1 -> partition 0)
    for gt_lesion in gt_lesions:
        gt_lesion_volume = np.zeros_like(gt)
        gt_lesion_slice = gt_lesion.get_coordinates()
        gt_lesion_volume[gt_lesion_slice] += gt[gt_lesion_slice]
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

def count_detections(dgraph=None, verbose=False):

    if verbose is True:
        print('Directed graph has {} nodes'.format(dgraph.number_of_nodes()))

    pred_lesion_nodes, gt_lesion_nodes = bipartite.sets(dgraph)

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_lesions = len(gt_lesion_nodes)

    # Count true positives and false negatives
    for gt_lesion_node in gt_lesion_nodes:
        incoming_edge_weights = []

        for pred_lesion_node in pred_lesion_nodes:
            # Examine edge weights
            edge_weight = dgraph[pred_lesion_node][gt_lesion_node]['weight']
            incoming_edge_weights.append(edge_weight)
            # Sanity check
            reverse_edge_weight = dgraph[gt_lesion_node][pred_lesion_node]['weight']
            assert(edge_weight == reverse_edge_weight)
        # Check the maximum weight
        max_weight = np.amax(np.array(incoming_edge_weights))
        if max_weight > 0: # Atleast one incoming edge with dice > 0
            true_positives += 1
        else:
            false_negatives += 1

    # Count false positives
    slices = []
    labels = []

    for pred_lesion_node in pred_lesion_nodes:
        outgoing_edge_weights = []

        for gt_lesion_node in gt_lesion_nodes:
            edge_weight = dgraph[pred_lesion_node][gt_lesion_node]['weight']
            outgoing_edge_weights.append(edge_weight)
            # Sanity check
            reverse_edge_weight = dgraph[gt_lesion_node][pred_lesion_node]['weight']
            assert(edge_weight == reverse_edge_weight)

        # Check maximum weight
        max_weight = np.amax(np.array(outgoing_edge_weights))
        slices.append(pred_lesion_node.get_coordinates())
        if max_weight == 0:
            false_positives += 1
            labels.append(1)
        else:
            labels.append(0)

    recall = true_positives/(true_positives + false_negatives)
    precision = true_positives/(true_positives + false_positives)

    lesion_counts_dict = {}
    lesion_counts_dict['slices'] = slices
    lesion_counts_dict['labels'] = labels
    lesion_counts_dict['recall'] = recall
    lesion_counts_dict['precision'] = precision
    lesion_counts_dict['true positives'] = true_positives
    lesion_counts_dict['false positives'] = false_positives
    lesion_counts_dict['false negatives'] = false_negatives
    lesion_counts_dict['true lesions'] = true_lesions

    return lesion_counts_dict


def filter_edges(dgraph):
    """

    Function to remove edges with zero weight (for better viz)

    """
    pred_lesion_nodes, gt_lesion_nodes = bipartite.sets(dgraph)

    # Create a dummy graph that has disconnected nodes for better visualization

    dgraph_viz = nx.DiGraph()

    # Create forward connections
    for pred_node in pred_lesion_nodes:
        weights = []
        for gt_node in gt_lesion_nodes:
            edge_weight = dgraph[pred_node][gt_node]['weight']
            weights.append(edge_weight)
            if edge_weight > 0:
                dgraph_viz.add_weighted_edges_from([(pred_node, gt_node, edge_weight)])

        max_weight = np.amax(np.array(weights))

        if max_weight == 0:
            dgraph_viz.add_node(pred_node) # False positive

    # Create backward connections
    for gt_node in gt_lesion_nodes:
        weights = []
        for pred_node in pred_lesion_nodes:
            edge_weight = dgraph[gt_node][pred_node]['weight']
            weights.append(edge_weight)
            if edge_weight > 0:
                dgraph_viz.add_weighted_edges_from([(gt_node, pred_node, edge_weight)])

        max_weight = np.amax(np.array(weights))

        if max_weight == 0:
            dgraph_viz.add_node(gt_node) # False negative

    return dgraph_viz


def visualize_lesion_correspondences(dgraph, fname=None):

    pred_lesion_nodes, gt_lesion_nodes = bipartite.sets(dgraph)

    dgraph_viz = filter_edges(dgraph)

    # Create a color map
    color_map = []
    for node in dgraph_viz:
        node_name = node.get_name()
        if "predicted" in node_name.lower():
            color_map.append('red')
        else:
            color_map.append('green')

    pos = nx.bipartite_layout(dgraph_viz, pred_lesion_nodes)

    nx.draw(dgraph_viz, pos=pos, node_color=color_map)

    # TODO: Add figure title and legend

    plt.savefig(fname)

    plt.close()




