import deepchem as dc
import os
import sys
os.environ['DGLBACKEND'] = 'pytorch'
import numpy as np
import pandas as pd
import dgl
import torch

from deepchem.feat.graph_data import GraphData
from deepchem.models import MPNNModel
from deepchem.feat import MolGraphConvFeaturizer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import math
import warnings
warnings.filterwarnings('ignore')
import copy
sys.path.append('../')
from utils import *
from featurizer_utils import *
from dc_molecule_feature_utils import *
from run_experiments import run_experiments
from rdkit import Chem
from itertools import product

class MPNNNormalizer:
    def __init__(self,feat_params,mol):
        self.node_feature_names = feat_params['feats']['node_feats']
        self.edge_feature_names = feat_params['feats']['edge_feats']
        self.node_feat_index = mol.node_feat_index
        self.edge_feat_index = mol.edge_feat_index
    
    def fit(self,X):
        nodes = []
        edges = []
        for g in X:
            for x in g.node_features:
                nodes.append(list(x))
            for e in g.edge_features:
                edges.append(list(e))
        self.node_mean = np.mean(nodes,axis=0)
        self.node_std = np.std(nodes,axis=0)
        self.node_max = np.max(nodes,axis=0)
        self.node_min = np.min(nodes,axis=0)
        self.edge_mean = np.mean(edges,axis=0)
        self.edge_std = np.std(edges,axis=0)
        self.edge_max = np.max(edges,axis=0)
        self.edge_min = np.min(edges,axis=0)

    def transform(self,X):
        for g in X:
            for i in range(len(g.node_features)):
                for feat in self.node_feature_names:
                    norm = self.node_feature_names[feat]
                    j = self.node_feat_index[feat]['start']
                    if(norm == 1):
                        g.node_features[i,j] = (g.node_features[i,j] - self.node_min[j]) / (self.node_max[j] - self.node_min[j])
                    elif(norm == 2):
                        g.node_features[i,j] = (g.node_features[i,j] - self.node_mean[j]) / self.node_std[j]
                    
            for i in range(len(g.edge_features)):
                for feat in self.edge_feature_names:
                    norm = self.edge_feature_names[feat]
                    j = self.edge_feat_index[feat]['start']
                    if(norm == 1):
                        g.edge_features[i,j] = (g.edge_features[i,j] - self.edge_min[j]) / (self.edge_max[j] - self.edge_min[j])
                    elif(norm == 2):
                        g.edge_features[i,j] = (g.edge_features[i,j] - self.edge_mean[j]) / self.edge_std[j]



