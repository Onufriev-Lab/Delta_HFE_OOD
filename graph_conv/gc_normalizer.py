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

class GCNormalizer:
    def __init__(self,feat_params,mol):
        self.node_feature_names = feat_params['feats']['node_feats']
        self.node_feat_index = mol.node_feat_index
    
    def fit(self,X):
        nodes = []
        edges = []
        for g in X:
            for x in g.atom_features:
                nodes.append(list(x))
        self.node_mean = np.mean(nodes,axis=0)
        self.node_std = np.std(nodes,axis=0)
        self.node_max = np.max(nodes,axis=0)
        self.node_min = np.min(nodes,axis=0)


    def transform(self,X):
        for g in X:
            for i in range(len(g.atom_features)):
                for feat in self.node_feature_names:
                    norm = self.node_feature_names[feat]
                    j = self.node_feat_index[feat]['start']
                    if(norm == 1):
                        g.atom_features[i,j] = (g.atom_features[i,j] - self.node_min[j]) / (self.node_max[j] - self.node_min[j])
                    elif(norm == 2):
                        g.atom_features[i,j] = (g.atom_features[i,j] - self.node_mean[j]) / self.node_std[j]
                    




