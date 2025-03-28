import deepchem as dc
import os
import sys
os.environ['DGLBACKEND'] = 'pytorch'
import numpy as np
import pandas as pd
import dgl
import torch

from deepchem.feat.mol_graphs import ConvMol
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

pqr_cols = ['field_name', 'atom_num', 'atom_name', 'residue', 'chain_id','X','Y','Z','charge','radius','atom_symbol']
decom_cols = ['atom1','atom2','decom']

node_feat_list = ['partial_charge','atom_gb','sum_pairwise_gb','sum_pairwise_gb_non_edge','sum_pairwise_gb_edge','inverse_born_radius','mbondi_radius','atom_identity','int_charge','hybridization','hydrogen_bonding','aromatic','degree','num_attached_H','implicit_valence','explicit_valence','electronegativity','flexibility']
edge_feat_list = ['distance','bonded','pairwise_gb','bond_type','same_ring','conjugated','stereo_config']


class GCMolecule:
    def __init__(self,df,paths,i,feat_params):
        self.node_feature_names = feat_params['feats']['node_feats']
        # self.edge_feature_names = feat_params['feats']['edge_feats']
        self.edge_cutoff_variable = feat_params['edges']['edge_cutoff_variable']
        self.cutoff_value = feat_params['edges'].get('edge_cutoff_value')
        self.implicit_h = feat_params['edges']['implicit_h']
        self.smiles = df.at[i,'smiles']
        self.df = df
        self.id = df.at[i,'id']
        self.rigid = df.at[i,'rigid']
        # print(self.id)
        self.paths = paths
        self.index = i
        self.pqr_df = trim_pqr_df(pd.read_csv(paths['pqr']+self.id+'.pqr',delim_whitespace=True,header=None,names=pqr_cols))
        self.adj,self.atom_symbol_list = get_prmtop_adj(paths['prmtop']+self.id+'.prmtop')
        self.decom_df = pd.read_csv(paths['decom_solv']+self.id+'.decom_solv',delim_whitespace=True,header=None,names=decom_cols)
        self.radii_series = pd.read_csv(paths['born_radii']+self.id+'.born_radii',delim_whitespace=True,header=None)[1]
        self.pqr_df,self.decom_df = atom_ids_subtract_one(self.pqr_df,self.decom_df)
        self.rd_mol = Chem.AddHs(Chem.MolFromSmiles(self.smiles))#GetAtomWithIdx, GetBondBetweenAtoms
        
        self.map_pqr_rd = get_index_map(self.adj,self.atom_symbol_list,self.rd_mol)
        if(self.implicit_h):
            self.rd_mol = Chem.MolFromSmiles(self.smiles)#GetAtomWithIdx, GetBondBetweenAtoms
        
        self.num_nodes,self.map_pqr_g = get_num_nodes(self.atom_symbol_list,self.implicit_h)
        if((self.edge_cutoff_variable == 'distance')):
            self.dist_adj = get_dist_adj(self.pqr_df)
        self.node_feat_index = {}
        self.edge_feat_index = {}
        self.g = None
        
    def create_graph(self):
        adj_list = self.__get_edges()
        # print(self.smiles,self.map_pqr_g,src,dst)
        if(len(adj_list) == 0):
            self.__molecule_with_no_edges_error()

        node_features=self.__get_graph_features()
        # print(len(adj_list),len(adj_list[0]))
        self.g = ConvMol(adj_list = adj_list,atom_features = node_features)
        
        # self.g.n_atoms = self.num_nodes
        
        # self.__add_graph_features()
        
        # self.n_feat = len(self.g.atom_features[0])

    def __get_edges(self):
        if(self.edge_cutoff_variable == 'fully_connected'):
            src,dst = get_decom_edges(self.decom_df,0,self.implicit_h,self.atom_symbol_list,self.map_pqr_g)
        elif(self.edge_cutoff_variable == 'pairwise_gb'):
            src,dst = get_decom_edges(self.decom_df,self.cutoff_value,self.implicit_h,self.atom_symbol_list,self.map_pqr_g)
        elif(self.edge_cutoff_variable == 'distance'):
            src,dst = get_distance_edges(self.dist_adj,self.cutoff_value,self.implicit_h,self.atom_symbol_list,self.map_pqr_g)
        elif(self.edge_cutoff_variable == 'bonds'):
            src,dst = get_bond_edges(self.adj,self.implicit_h,self.atom_symbol_list,self.map_pqr_g)
        adj_list = self.__get_adj_list(src,dst)
        return adj_list

    def __get_adj_list(self,src,dst):
        adj_list = [[] for n in range(self.num_nodes)]
        for s,d in zip(src,dst):
            adj_list[s].append(d)
        return adj_list
    
    def __get_graph_features(self):
        num_nodes = self.num_nodes
        # num_edges = self.num_edges
        node_features = [[] for i in range(num_nodes)]
        # edge_features = [[] for i in range(num_edges)]
        # print(self.id,self.smiles,self.map_pqr_g)
        i = 0
        feat_index = 0
        first_node = 1
        for i in range(len(self.atom_symbol_list)):
            if((not self.implicit_h) or (self.atom_symbol_list[i] != 'H')):
                for node_feat in self.node_feature_names:
                    g_i = self.map_pqr_g[i]
                    feat = self.__get_atom_node_feat(i,node_feat)
                    node_features[g_i]+=feat
                    if(first_node):
                        self.node_feat_index[node_feat] = {'start':feat_index,'end':feat_index+len(feat)}
                        feat_index += len(feat)
                first_node = 0
        return np.array(node_features)
        
            
    def __get_atom_node_feat(self,i,feat):
        rd_i = self.map_pqr_rd[i]
        atom = self.rd_mol.GetAtomWithIdx(rd_i)
        if(feat == 'partial_charge'):
            return get_atom_pqr_partial_charge(i,self.pqr_df)
        elif(feat == 'atom_gb'):
            return get_atom_gb(i,self.decom_df,self.num_nodes)
        elif(feat == 'inverse_born_radius'):
            return get_atom_inverse_born_radius(i,self.radii_series)
        elif(feat == 'mbondi_radius'):
            return get_atom_mbondi_radius(i,self.pqr_df)
        elif(feat == 'sum_pairwise_gb'):
            return get_atom_sum_pairwise_gb(i,self.decom_df,self.g.num_nodes)
        elif(feat == 'atom_identity'):
            return get_atom_type_one_hot(atom)
        elif(feat == 'int_charge'):
            return get_atom_formal_charge(atom)
        elif(feat == 'hybridization'):
            return get_atom_hybridization_one_hot(atom)
        elif(feat == 'hydrogen_bonding'):
            return get_atom_hydrogen_bonding_one_hot(atom,construct_hydrogen_bonding_info(self.rd_mol))
        elif(feat == 'aromatic'):
            return get_atom_is_in_aromatic_one_hot(atom)
        elif(feat == 'degree'):
            return get_atom_total_degree_one_hot(atom)
        elif(feat == 'num_attached_H'):
            return get_atom_total_num_Hs_one_hot(atom)
        elif(feat == 'implicit_valence'):
            return get_atom_implicit_valence_one_hot(atom)
        elif(feat == 'explicit_valence'):
            return get_atom_explicit_valence_one_hot(atom)
        elif(feat == 'electronegativity'):
            return get_atom_electronegativity(atom)
        elif(feat == 'flexibility'):
            return [self.rigid]
        self.__error_not_correct_feat_name(feat,True)

    def __error_not_correct_feat_name(self,name,node):
        if(node):
            raise ValueError(f'{name} is not an accepted node feature name. Please only use names from the following list: {node_feat_list}')
        if(not node):
            raise ValueError(f'{name} is not an accepted edge feature name. Please only use names from the following list: {edge_feat_list}')

    def __molecule_with_no_edges_error(self):
        raise ValueError(f'Molecule {self.id} with smiles "{self.smiles}" has no edges. Please remove this molecule from the dataset or include H explicitly')

class GCFeaturizer:
    def __init__(self,feat_params):
        self.feat_params = feat_params

    def featurize(self,df,dataset_name):
        df = df.reset_index(drop=True)
        paths = get_input_file_paths(dataset_name)
        X = []
        Molecules = []
        for i in range(len(df)):
            g = GCMolecule(df,paths,i,self.feat_params)
            g.create_graph()
            X.append(g.g)
            Molecules.append(g)
        return X,Molecules












    