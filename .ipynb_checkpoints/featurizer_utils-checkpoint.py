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
from run_experiments import run_experiments
from dc_molecule_feature_utils import *
from rdkit import Chem
from itertools import product



def get_edge_bond_type(bond):
    if(bond):
        return get_bond_type_one_hot(bond)
    else:
        return [0.,0.,0.,0.]
        
def get_edge_same_ring(bond):
    if(bond):
        return get_bond_is_in_same_ring_one_hot(bond)
    else:
        return [0.]
        
def get_edge_conjugated(bond):
    if(bond):
        return get_bond_is_conjugated_one_hot(bond)
    else:
        return [0.]
        
def get_edge_stereo_config(bond):
    if(bond):
        return get_bond_stereo_one_hot(bond)
    else:
        return [0.,0.,0.,0.,0.]

def get_edge_inverse_distance(i,j,dist_adj):
    return [1./dist_adj[i,j]]
    # return [dist_adj[i,j]]

def get_edge_bonded(i,j,adj):
    return [adj[i,j]]

def get_edge_pairwise_gb(i,j,decom_df):
    mn = min(i,j)
    mx = max(i,j)
    # print(decom_df[(decom_df.atom1 == mn) & (decom_df.atom2 == mx)]['decom'])
    return [decom_df[(decom_df.atom1 == mn) & (decom_df.atom2 == mx)]['decom'].sum()]

def get_atom_pqr_partial_charge(i,pqr_df):
    return [pqr_df['charge'][i]]

def get_atom_mbondi_radius(i,pqr_df):
    return [pqr_df['radius'][i]]

def get_atom_gb(i,decom_df,num_nodes):
    return [decom_df[(decom_df.atom1 == i) & (decom_df.atom2 == i)]['decom'].sum()]

def get_atom_sum_pairwise_gb(i,decom_df,num_nodes):
    return [decom_df[(decom_df.atom1 == i) ^ (decom_df.atom2 == i)]['decom'].sum()]

def get_atom_inverse_born_radius(i,radii_series):
    return [radii_series[i]]

def get_num_nodes(atom_symbols,implicit_h):
    map = {}
    if(implicit_h):
        n = 0
        i = 0
        for atom in atom_symbols:
            if(atom != 'H'):
                map[i] = n
                n+=1
            i+=1
        return n,map
    else:
        for i in range(len(atom_symbols)):
            map[i] = i
        return len(atom_symbols),map
        

def get_dist_adj(pqr_df):
    coordinates = pqr_df.iloc[:, 5:8].values
    num_nodes = len(coordinates)
    adj_matrix = np.zeros((num_nodes, num_nodes))
    
    for i in range(num_nodes):
        distances = np.linalg.norm(coordinates - coordinates[i], axis=1)
        adj_matrix[i, :] = distances
        adj_matrix[:, i] = distances

    return adj_matrix

def get_distance_edges(dist_adj,threshold,implicit_h,atom_symbols,map_pqr_g):
    src,dst = [],[]
    num_atoms = len(dist_adj)
    for i in range(num_atoms):
        if((not implicit_h) or (atom_symbols[i] != 'H')):
            for j in range(i+1,num_atoms):
                if((dist_adj[i,j] <= threshold) and ((atom_symbols[j] != 'H') or (not implicit_h))):
                    src.append(map_pqr_g[i])
                    dst.append(map_pqr_g[j])
                    src.append(map_pqr_g[j])
                    dst.append(map_pqr_g[i])
    return src,dst

def get_decom_edges(decom_df,threshold,implicit_h,atom_symbols,map_pqr_g):
    src,dst = [],[]
    for i,row in decom_df.iterrows():
        if(row['atom1']!=row['atom2']):
            i = int(row['atom1'])
            j = int(row['atom2'])
            if((not implicit_h) or ((atom_symbols[i] != 'H') and (atom_symbols[j] != 'H'))):
                if(abs(row['decom']) >= threshold):
                    src.append(map_pqr_g[i])
                    dst.append(map_pqr_g[j])
                    src.append(map_pqr_g[j])
                    dst.append(map_pqr_g[i])
    return src,dst

def get_bond_edges(adj,implicit_h,atom_symbols,map_pqr_g):
    src,dst = [],[]
    num_atoms = len(adj)
    for i in range(num_atoms):
        if((not implicit_h) or (atom_symbols[i] != 'H')):
            for j in range(i+1,num_atoms):
                if((adj[i,j] == 1) and ((atom_symbols[j] != 'H') or (not implicit_h))):
                    src.append(map_pqr_g[i])
                    dst.append(map_pqr_g[j])
                    src.append(map_pqr_g[j])
                    dst.append(map_pqr_g[i])
    return src,dst

def atom_ids_subtract_one(pqr_df,decom_df):
    decom_df['atom1'] -= 1
    decom_df['atom2'] -= 1
    pqr_df['atom_num'] = pd.to_numeric(pqr_df['atom_num']) - 1
    # pqr_df['atom_num'] -= 1
    return pqr_df,decom_df

def vf2_match(G1, G2):
    def is_feasible(v1, v2, mapping):
        # Check if the nodes are compatible (e.g., same atom type)
        if G1['nodes'][v1] != G2['nodes'][v2]:
            return False

        # Check adjacency consistency
        for u1, u2 in mapping.items():
            if (G1['adj'][v1][u1] != G2['adj'][v2][u2]) or (G1['adj'][u1][v1] != G2['adj'][u2][v2]):
                return False
        return True

    def match_aux(mapping):
        if len(mapping) == len(G1['nodes']):
            return mapping

        for v1 in range(len(G1['nodes'])):
            if v1 not in mapping:
                for v2 in range(len(G2['nodes'])):
                    if v2 not in mapping.values() and is_feasible(v1, v2, mapping):
                        mapping[v1] = v2
                        result = match_aux(mapping)
                        if result:
                            return result
                        del mapping[v1]
                return None

    return match_aux({})

def vf2_subgraph_match_matrix(smallG, largeG):
    def is_feasible(v1, v2, mapping):
        # Check if the nodes are compatible (e.g., same atom type)
        if smallG['nodes'][v1] != largeG['nodes'][v2]:
            return False

        # Check adjacency consistency
        for u1, u2 in mapping.items():
            if smallG['adj'][v1][u1] != largeG['adj'][v2][u2]:
                return False
        return True

    def match_aux(mapping):
        if len(mapping) == len(smallG['nodes']):
            return mapping

        for v1 in range(len(smallG['nodes'])):
            if v1 not in mapping:
                for v2 in range(len(largeG['nodes'])):
                    if v2 not in mapping.values() and is_feasible(v1, v2, mapping):
                        mapping[v1] = v2
                        result = match_aux(mapping)
                        if result:
                            return result
                        del mapping[v1]
                return None

    return match_aux({})

def get_index_map(adj,atom_symbol_list,rd_mol):
    atom_objects = rd_mol.GetAtoms()
    rd_atoms = []
    for j in range(len(atom_objects)):
        rd_atoms.append(atom_objects[j].GetSymbol())
    rd_adj = np.array(Chem.GetAdjacencyMatrix(rd_mol))
    top = {'nodes':atom_symbol_list,'adj':adj}
    rd = {'nodes':rd_atoms,'adj':rd_adj}
    # print(sorted(top['nodes']),'\n',sorted(rd['nodes']))
    mapping = vf2_match(top,rd)
    return mapping

def get_prmtop_adj(prmtop_file):
    with open(prmtop_file,'r') as f:
        lines = f.readlines()
    line_length = len(lines[1])

    if('-' not in prmtop_file):
        z = '%FLAG ATOMIC_NUMBER'
        z_index = lines.index(z + ' '*(line_length - len(z) - 1) + '\n') + 2
        z = lines[z_index].split()
        while (not lines[z_index + 1].startswith('%')):
            z_index += 1
            z += lines[z_index].split()
        num_nodes = len(z)
    else:
        z_str = '%FLAG AMBER_ATOM_TYPE'
        if(line_length < 70):#for improperly formatted prmtop files
            z_index = lines.index(z_str + '\n') + 2
        else:
            z_index = lines.index(z_str + ' '*(line_length - len(z_str) - 1) + '\n') + 2
        z = []
        for a in lines[z_index].split():
            z += a[0].upper()
        while (not lines[z_index + 1].startswith('%')):
            z_index += 1
            for a in lines[z_index].split():
                z += a[0].upper()
        num_nodes = len(z)

    
    bond_h_str = '%FLAG BONDS_INC_HYDROGEN'
    bond_no_h_str = '%FLAG BONDS_WITHOUT_HYDROGEN'
    h_index = lines.index(bond_h_str + ' '*(line_length - len(bond_h_str) - 1) + '\n') + 2
    no_h_index = lines.index(bond_no_h_str + ' '*(line_length - len(bond_no_h_str) - 1) + '\n') + 2
    h_bonds = lines[h_index].split()
    while (not lines[h_index + 1].startswith('%')):
        h_index += 1
        h_bonds += lines[h_index].split()

    no_h_bonds = lines[no_h_index].split()
    while (not lines[no_h_index + 1].startswith('%')):
        no_h_index += 1
        no_h_bonds += lines[no_h_index].split()

    adj_matrix = np.zeros((num_nodes,num_nodes),dtype=np.int8)

    # print(adj_matrix)
    for i in range(0,len(h_bonds),3):
        a = int(int(h_bonds[i])/3)
        b = int(int(h_bonds[i+1])/3)
        # print(a,b)
        adj_matrix[a,b] = 1
        adj_matrix[b,a] = 1
        
    for i in range(0,len(no_h_bonds),3):
        a = int(int(no_h_bonds[i])/3)
        b = int(int(no_h_bonds[i+1])/3)
        # print(a,b)
        adj_matrix[a,b] = 1
        adj_matrix[b,a] = 1

    symbols = []
    if('-' not in prmtop_file):
        for a in z:
            symbols.append(Chem.Atom(int(a)).GetSymbol())
    else:
        symbols = z
    return adj_matrix,symbols

def get_input_file_paths(name):
    if(name == 'freesolv' or name == 'train' or name == 'test' or name == 'valid' or name == 'train_val'):
        paths = {
            'pqr':'../../physics/PQR/',
            'prmtop':'../../physics/CRDTOP/',
            'crd':'../../physics/CRDTOP/',
            'born_radii':'../../physics/born_radii_freesolv_GBNSR6/mbondi/',
            'decom_solv':'../../physics/decom_solv_freesolv_GBNSR6/mbondi/',
        }
    elif(name == 'amino'):
        paths = {
            'pqr':'../../physics/amino_44/pqr/',
            'prmtop':'../../physics/amino_44/top/',
            'crd':'../../physics/amino_44/crd/',
            'born_radii':'../../physics/amino_44/born_radii/',
            'decom_solv':'../../physics/amino_44/decom/',
        }
    return paths

def same_atomic_symbol(a,b):
    return a.upper() == b.upper()

def trim_pqr_df(pqr_df):
    pqr_df = pqr_df[pqr_df.field_name == 'ATOM']
    return pqr_df.reset_index(drop=True)