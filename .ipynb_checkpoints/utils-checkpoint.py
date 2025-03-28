import numpy as np
import pandas as pd
import sys
import glob
import deepchem as dc
# from custom_mpnn_model.custom_mpnn import CustomMPNN
# from custom_mpnn_model.dual_mpnn import DualMPNN
# from custom_mpnn_model.node_pred_mpnn import NodePredMPNN
# from custom_mpnn_model.pairwise_pred_mpnn import PairwisePredMPNN
import os
import json
import re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
import warnings
import shutil
import torch
warnings.filterwarnings('ignore')
from itertools import product
abbr = {
    'epochs':'ep',
    'use_chirality': 'chiral',
    'use_partial_charge': 'charge',
    'es_over_patience':'opat',
    'es_overfitting':'over',
    'es_conv_patience':'cpat',
    'es_min_delta':'del',
    'batch_size':'batch',
    'learning_rate':'lr',
    'node_out_feats':'node_out',
    'edge_hidden_feats':'edge_hid',
    'num_step_message_passing':'num_mp',
    'num_step_set2set':'step_s2s',
    'num_layer_set2set':'layer_s2s',
    'graph_conv_layers':'gc_layers',
    'dropout':'drop',
    'batch_normalize':'b_norm',
    'dense_layer_size':'dense',
    'node_feats':'node',
    'edge_feats':'edge',
    'edge_cutoff_variable':'var',
    'edge_cutoff_value':'cut',
    'fully_connected':'fc',
    'partial_charge':'c',
    'distance':'r',
    'inverse_born_radius':'br',
    'mbondi_radius':'mr',
    'pairwise_gb':'gb',
    'atom_gb':'gb',
    'sum_pairwise_gb':'sgb',
    'sum_pairwise_gb_non_edge':'sngb',
    'sum_pairwise_gb_edge':'segb',
    'atom_identity':'z',
    'int_charge':'ic',
    'hybridization':'hy',
    'hydrogen_bonding':'hb',
    'aromatic':'ar',
    'degree':'de',
    'num_attached_H':'nh',
    'implicit_valence':'iv',
    'explicit_valence':'ev',
    'bonded':'b',
    'bond_type':'bt',
    'same_ring':'sr',
    'conjugated':'co',
    'stereo_config':'sc',
    'electronegativity':'el',
    'flexibility':'fl'
}

def split_settings_column_summary_df(df):
    df[['feats','hyper']] = df['settings'].str.split('/',n=1,expand=True)
    df["hyper"] = df["hyper"].str.strip("/")
    n = df.hyper.str.count('-')[0]
    split_pattern = r'(?<=\D)_(?=\d)'
    for i in range(n):
        df[['temp','hyper']] = df['hyper'].str.split('-',n=1,expand=True)
        split = re.split(split_pattern,df.at[0,'temp'])
        name = split[0]
        df[name] = 0
        for j in range(len(df)):
            split = re.split(split_pattern,df.at[j,'temp'])
            df.at[j,name] = underscore_to_numeric(split[1])
    split = re.split(split_pattern,df.at[0,'hyper'])
    name = split[0]
    df[name] = 0
    for j in range(len(df)):
        split = re.split(split_pattern,df.at[j,'hyper'])
        df.at[j,name] = underscore_to_numeric(split[1])
    df.drop(columns=['temp','hyper','settings'],inplace=True)
    return df
    
def underscore_to_numeric(str):
    # print(str,str.replace('_','.'),float(str.replace('_','.')))
    return float(str.replace('_','.'))
    
    

def get_param_combos(folder):
    f = open(folder+'feat_params.json')
    feat_params_dict = json.load(f)
    f = open(folder+'hyper_params.json')
    hyper_params_dict = json.load(f)
    feat_param_combos = list(product(*feat_params_dict.values()))
    hyper_combos = list(product(*hyper_params_dict.values()))
    return feat_param_combos,hyper_combos,feat_params_dict,hyper_params_dict

def delete_graph_conv_checkpoints(folder,ckpt_to_keep):
    for file_name in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, file_name)):
            if((file_name != 'checkpoint') and (file_name != ckpt_to_keep+'.index') and (file_name != ckpt_to_keep+'.data-00000-of-00001')):
                os.remove(os.path.join(folder, file_name))

def delete_mpnn_checkpoints(folder,ckpt_to_keep):
    for file_name in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, file_name)):
            if(file_name != ckpt_to_keep):
                os.remove(os.path.join(folder, file_name))

def delete_all_checkpoints(folders):
    for folder in folders:
        for phys_model_dir in os.listdir(folder):
            phys_folder = os.path.join(folder,phys_model_dir)
            if os.path.isdir(phys_folder):
                for model_dir in os.listdir(phys_folder):
                    model_folder = os.path.join(phys_folder,model_dir)
                    if os.path.isdir(model_folder):
                        for chkp_dir in os.listdir(model_folder):
                            ch_folder = os.path.join(model_folder,chkp_dir)
                            if os.path.isdir(ch_folder):
                                shutil.rmtree(ch_folder)
                        

def print_features_with_names(mol):
    print(mol.id,mol.smiles)
    print()
    print(f'Nodes ({len(mol.node_feature_names)} feature names, {len(mol.g.node_features[0])} total features):')
    for i in range(len(mol.atom_symbol_list)):
        g_i = mol.map_pqr_g.get(i)
        if(g_i is not None):
            print(f'{mol.atom_symbol_list[i]} pqr_index={i} graph_index={g_i}')
            for name in mol.node_feature_names:
                indexes = mol.node_feat_index
                print(f'    {name}: {mol.g.node_features[g_i][indexes[name]["start"]:indexes[name]["end"]]}')
    eindex = 0
    print(f'\nEdges ({len(mol.edge_feature_names)} feature names, {len(mol.g.edge_features[0])} total features):')
    for i in range(len(mol.atom_symbol_list)):
        for j in range(i+1,len(mol.atom_symbol_list)):
            if((not mol.implicit_h) or ((mol.atom_symbol_list[i] != 'H') and (mol.atom_symbol_list[j] != 'H'))):
                g_i = mol.map_pqr_g[i]
                g_j = mol.map_pqr_g[j]
                if(((mol.g.edge_index[0] == g_i) & (mol.g.edge_index[1] == g_j)).any().item()):
                    print(f'({mol.atom_symbol_list[i]},{mol.atom_symbol_list[j]}) pqr_index=({i},{j}) graph_index=({g_i},{g_j})')
                    for name in mol.edge_feature_names:
                        name = mol.edge_feature_names[k]
                        indexes = mol.edge_feat_index
                        print(f'    {name}: {mol.g.edge_features[eindex][indexes[name]["start"]:indexes[name]["end"]]}')
                    eindex+=2
            

def print_model(iter,final_epoch,loss,dataset_names,rmse):
    temp = f'Model {iter}: epochs {final_epoch[0]}, criterion: {final_epoch[1]}, loss {loss:.3f}, RMSE: '
    for i in range(len(dataset_names)):
        temp += f'{dataset_names[i]} {rmse[i]:.3f}, '
    print(temp[:-2])

def underlined(str):
    return '\033[4m' + str + '\033[0m'

def print_data(dataset_names,phys_rmse,ml_mean,ml_std,ml_ens,rmse_df,underlined=False):
    physics_str = f'physics model '
    phys_ml_str = f'physics + ml  '
    ensemble_str = f'ensemble ml   '
    col_names = 14*' '
    for i in range(len(dataset_names)):
        physics_str = physics_str + f'| {phys_rmse[i]:.3f}          '
        phys_ml_str = phys_ml_str + f'| {ml_mean[i]:.3f} ± {ml_std[i]:.3f}  '
        ensemble_str = ensemble_str + f'| {ml_ens[i]:.3f}          '
        col_names = col_names + f'| {dataset_names[i][:-4]}' + (15-len(dataset_names[i][:-4]))*' '
    print(f'Total models: {len(rmse_df)}, mean number of epochs: {np.mean(rmse_df.epochs):.1f} ± {np.std(rmse_df.epochs):.1f}')
    if(underlined):
        print(underlined(col_names))
        print(physics_str)
        print(phys_ml_str)
        print(underlined(ensemble_str))
    else:
        print(col_names)
        print(physics_str)
        print(phys_ml_str)
        print(ensemble_str)

def print_summary_data(folder):
    dirs = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)) and not d.startswith('.')]
    dataset_names = [d for d in os.listdir(folder+dirs[0]+'/') if os.path.isfile(os.path.join(folder+dirs[0]+'/', d)) and d.endswith('.csv') and not d.startswith('convergence')]
    rmse_df = pd.read_csv(folder+'rmse.csv')
    len_datasets = []
    phys_rmse, ml_mean,ml_std,ml_ens = [],[],[],[]
    for dataset in dataset_names:
        df = pd.read_csv(folder+dirs[0]+'/'+dataset)
        len_datasets.append(len(df))
        # phys_rmse.append(rmsd(df.expt,df.physics))
        phys_rmse.append(md(df.physics,df.expt))

    ml_mean,ml_std,ml_ens = [],[],[]
    for dataset in dataset_names:
        dataset = dataset[:-4]
        ml_mean.append(np.mean(rmse_df[dataset]))
        ml_std.append(np.std(rmse_df[dataset]))
    ml_ens = get_ensemble_results(folder,True)
    print_data(dataset_names,phys_rmse,ml_mean,ml_std,ml_ens,rmse_df)

def get_ensemble_results(folder,save = False):
    dirs = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)) and not d.startswith('.')]
    dataset_names = [d for d in os.listdir(folder+dirs[0]+'/') if os.path.isfile(os.path.join(folder+dirs[0]+'/', d)) and d.endswith('.csv') and not d.startswith('convergence')]
    ml_ens = []
    for i in range(len(dataset_names)):
        df = pd.read_csv(folder+dirs[0]+'/'+dataset_names[i])
        array = np.zeros((len(df),len(dirs)))
        for j in range(len(dirs)):
            temp = pd.read_csv(folder+dirs[j]+'/'+dataset_names[i])
            array[:,j] = temp.physics_ml
        df.physics_ml = np.mean(array,axis=1)
        df.ml = df.physics_ml - df.physics
        # ml_ens.append(rmsd(df.expt,df.physics_ml))
        ml_ens.append(md(df.physics_ml,df.expt))
        if(save):
            df.to_csv(folder+'ens_'+dataset_names[i],index=False)
        
    return ml_ens

def get_physics_model_name(model):
    namemap = {'tip3p':'TIP3P','cha':'CHA-GB','zap9':'GBNSR6 (ZAP9)','mbondi':'GBSNR6 (mbondi)','igb5':'IGB5','asc':'AASC','null':'ML alone'}
    if(model == 'null'):
        return 'Null'
    return namemap.get(model) or model

def print_all_physics_models(folder,with_model_names=True):
    dirs = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)) and not d.startswith('.')]
    for d in dirs:
        if(with_model_names):
            print(get_physics_model_name(d))
        print_summary_data(folder+d+'/')

def get_max_epochs(hyper_params,model_type):
    default_epochs = [100,500]
    if(using_early_stopping(hyper_params)):
        default_epochs = [200,1000]
    if(model_type == 'mpnn'):
        return (hyper_params.get('epochs') or default_epochs[0])
    elif(model_type == 'graphconv'):
        return (hyper_params.get('epochs') or default_epochs[1])
    elif(model_type == 'custom_mpnn' or model_type == 'dual_mpnn' or model_type == 'node_pred_mpnn' or model_type == 'pairwise_pred_mpnn'):
        return (hyper_params.get('epochs') or default_epochs[0])

def using_early_stopping(hyper_params):
    return hyper_params.get('es_min_delta')

def create_folder_name(folder_base,params,feat_type = 'default'):
    folder = folder_base
    if(feat_type == 'custom_mpnn' or feat_type == 'custom_graphconv'):
        for p in params:
            folder = folder + (abbr.get(p) or p) + '-'
            for q in params[p]:
                folder += (abbr.get(q) or q) + '_'
                if((type(params[p][q]) is list) or (isinstance(params[p][q],dict))):
                    for a in params[p][q]:
                        folder+=(abbr.get(a) or str(a)) + '-'
                else:
                    folder+=str(abbr.get(params[p][q]) or params[p][q]).replace('.', '_') + '-'
    else:
        for p in params:
            folder = folder + (abbr.get(p) or p) + '_' + str(params[p]).replace('.', '_') + '-'
    folder = folder[:-1] + '/'
    #modify for custom feats
    return folder

def get_full_folder_name(folder_base,feat_params,hyper_params,abbr):
    folder = create_folder_name(folder_base,feat_params,abbr)
    folder = create_folder_name(folder,hyper_params,abbr)
    return folder

def rmsd(a, b=None): # root mean square deviation
    if(type(b) == type(None)):
        b = np.zeros(len(a))
    return np.sqrt(np.mean(np.power(np.array(a)-np.array(b), 2)))

def md(a, b=None): # mean devialtion
    if(type(b) == type(None)):
        b = np.zeros(len(a))
    return np.mean(np.array(a)-np.array(b))

def ormsd(p, a, b=None): # rnsd of p fraction of outliers
    if(type(b) == type(None)):
        b = np.zeros(len(a))
    o = np.sort(np.abs(np.array(a)-np.array(b)))[int((1.0-p)*len(a)):]
    return rmsd(o)

def subSample(a, i):# index sampling
    c = []
    for j in i:
        c.append(a[j])
    return np.array(c)

def psuedoScramble(v, a = None, bins=10):# evenly splits a based on v
    if(a == None):
        a = np.arange(len(v))
    bin=[]
    v = np.argsort(v)
    binw = len(v)/float(bins)
    for i in range(bins):
        v[int(binw*i):int(binw*i+binw)] = np.random.permutation(v[int(binw*i):int(binw*i+binw)])
    return subSample(a, v)


def get_current_graph_conv_checkpoint_num(model_folder):
    #unnecessary function now
    checkpoint_file = model_folder + 'checkpoints/checkpoint'
    with open(checkpoint_file, 'r') as file:
        lines = file.readlines()
    num = [i for i in lines[0] if i.isdigit()]
    num = int("".join(num))
    return num

def create_model(datasets,model_type,hyper_params,folder,device):
    checkpoint_folder = folder+'checkpoints'
    if(model_type == 'mpnn'):
        model = dc.models.torch_models.MPNNModel(n_tasks = 1,model_dir = checkpoint_folder, 
                                                 number_atom_features = datasets[0].X[0].num_node_features, 
                                                 number_bond_features = datasets[0].X[0].num_edges_features,
                                                 batch_size=(hyper_params.get('batch_size') or 100), 
                                                 learning_rate=(hyper_params.get('learning_rate') or 0.001),
                                                 node_out_feats=(hyper_params.get('node_out_feats') or 64),
                                                 edge_hidden_feats=(hyper_params.get('edge_hidden_feats') or 128),
                                                 num_step_message_passing=(hyper_params.get('num_step_message_passing') or 3),
                                                 num_step_set2set=(hyper_params.get('num_step_set2set') or 6),
                                                 num_layer_set2set=(hyper_params.get('num_layer_set2set') or 3),
                                                 device = device
                                                )
    elif(model_type == 'custom_mpnn'):
        model = CustomMPNN(n_tasks = 1,model_dir = checkpoint_folder, 
                                                 number_atom_features = datasets[0].X[0].num_node_features, 
                                                 number_bond_features = datasets[0].X[0].num_edges_features,
                                                 batch_size=(hyper_params.get('batch_size') or 100), 
                                                 learning_rate=(hyper_params.get('learning_rate') or 0.001),
                                                 node_out_feats=(hyper_params.get('node_out_feats') or 64),
                                                 edge_hidden_feats=(hyper_params.get('edge_hidden_feats') or 128),
                                                 num_step_message_passing=(hyper_params.get('num_step_message_passing') or 3),
                                                 num_step_set2set=(hyper_params.get('num_step_set2set') or 6),
                                                 num_layer_set2set=(hyper_params.get('num_layer_set2set') or 3),
                                                 weight_decay = (hyper_params.get('weight_decay') or 0.0),
                                                 device = device
                                                )
    elif(model_type == 'node_pred_mpnn'):
        model = NodePredMPNN(n_tasks = 1,model_dir = checkpoint_folder, 
                                                 number_atom_features = datasets[0].X[0].num_node_features, 
                                                 number_bond_features = datasets[0].X[0].num_edges_features,
                                                 batch_size=(hyper_params.get('batch_size') or 100), 
                                                 learning_rate=(hyper_params.get('learning_rate') or 0.001),
                                                 node_out_feats=(hyper_params.get('node_out_feats') or 64),
                                                 edge_hidden_feats=(hyper_params.get('edge_hidden_feats') or 128),
                                                 num_step_message_passing=(hyper_params.get('num_step_message_passing') or 3),
                                                 num_step_set2set=(hyper_params.get('num_step_set2set') or 6),
                                                 num_layer_set2set=(hyper_params.get('num_layer_set2set') or 3),
                                                 weight_decay = (hyper_params.get('weight_decay') or 0.0),
                                                 device = device
                                                )
    elif(model_type == 'pairwise_pred_mpnn'):
        model = PairwisePredMPNN(n_tasks = 1,model_dir = checkpoint_folder, 
                                                 number_atom_features = datasets[0].X[0].num_node_features, 
                                                 number_bond_features = datasets[0].X[0].num_edges_features,
                                                 batch_size=(hyper_params.get('batch_size') or 100), 
                                                 learning_rate=(hyper_params.get('learning_rate') or 0.001),
                                                 node_out_feats=(hyper_params.get('node_out_feats') or 64),
                                                 edge_hidden_feats=(hyper_params.get('edge_hidden_feats') or 128),
                                                 num_step_message_passing=(hyper_params.get('num_step_message_passing') or 3),
                                                 num_step_set2set=(hyper_params.get('num_step_set2set') or 6),
                                                 num_layer_set2set=(hyper_params.get('num_layer_set2set') or 3),
                                                 device = device
                                                )
    elif(model_type == 'dual_mpnn'):
        model = DualMPNN(n_tasks = 1,model_dir = checkpoint_folder, 
                                                 number_atom_features = datasets[0].X[0].chem_g.num_node_features, 
                                                 number_bond_features = datasets[0].X[0].chem_g.num_edges_features,
                                                 number_atom_features_phys = datasets[0].X[0].phys_g.num_node_features, 
                                                 number_bond_features_phys = datasets[0].X[0].phys_g.num_edges_features,
                                                 batch_size=(hyper_params.get('batch_size') or 100), 
                                                 learning_rate=(hyper_params.get('learning_rate') or 0.001),
                                                 node_out_feats=(hyper_params.get('node_out_feats') or 64),
                                                 edge_hidden_feats=(hyper_params.get('edge_hidden_feats') or 128),
                                                 num_step_message_passing=(hyper_params.get('num_step_message_passing') or 3),
                                                 num_step_set2set=(hyper_params.get('num_step_set2set') or 6),
                                                 num_layer_set2set=(hyper_params.get('num_layer_set2set') or 3),
                                                 device = device
                                                )
    elif(model_type == 'graphconv'):
        model = dc.models.GraphConvModel(n_tasks=1, model_dir = checkpoint_folder,mode='regression', 
                                                 batch_size=(hyper_params.get('batch_size') or 100), 
                                                 learning_rate=(hyper_params.get('learning_rate') or 0.001),
                                                 graph_conv_layers=(hyper_params.get('graph_conv_layers') or [53,38]),
                                                 dropout=(hyper_params.get('dropout') or 0.4),
                                                 batch_normalize=(hyper_params.get('batch_normalize') or False),
                                                 dense_layer_size=(hyper_params.get('dense_layer_size') or 27),
                                        )
    else:
        print('Error: please enter "mpnn", "custom_mpnn", or "graphconv" as model_type')
    return model

class ZScoreNormalizer:
    def __init__(self):
        self.mean = 0
        self.std = 0
    
    def fit(self, data):
        """Fit the normalizer to the data, calculating mean and standard deviation."""
        data = np.array(data)
        self.mean = data.mean()
        self.std = data.std()
        # self.mean = sum(data) / len(data)
        # self.std = (sum((x - self.mean) ** 2 for x in data) / len(data)) ** 0.5

    def transform(self, data):
        """Apply Z-score normalization to the data."""
        if self.std == 0:
            raise ValueError("Standard deviation is zero, cannot perform normalization.")
        return np.array([(x - self.mean) / self.std for x in data])

    def untransform(self, data):
        """Revert Z-score normalization."""
        return np.array([(x * self.std) + self.mean for x in data])