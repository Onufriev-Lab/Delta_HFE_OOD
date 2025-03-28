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
# from custom_mpnn_model.custom_mpnn import CustomMPNN
import math

import copy
from utils import *
from save_data import *
from itertools import product

def model_is_overfitted(epoch,rmse_conv,over,opat):
    e = epoch-2
    for i in range(e,e-opat,-1):
        if(((rmse_conv[1][i] - rmse_conv[0][i])/(rmse_conv[1][i])) < over):
            return False
    return True
    #below for absolute threshold
    # e = epoch-2
    # for i in range(e,e-opat,-1):
    #     if((rmse_conv[1][i] - rmse_conv[0][i]) < over):
    #         return False
    # return True

def model_is_converged(epoch,rmse_conv,delta,cpat):
    e = epoch - 2
    ref = rmse_conv[1][e-cpat]
    for i in range(e,e-cpat,-1):
        if((ref - rmse_conv[1][i]) > delta):
            return False
    return True
        
def should_stop_training(epoch,max_epochs,hyper_params,rmse_conv,final_epoch):
    if(epoch > max_epochs):
        final_epoch[:] = [epoch-1,'max_epoch']
        return True
    elif(using_early_stopping(hyper_params)):
        over = hyper_params['es_overfitting']
        opat = hyper_params['es_over_patience']
        delta = hyper_params['es_min_delta']
        cpat = hyper_params['es_conv_patience']
        if(len(rmse_conv[0]) > max(opat,cpat)):
            if(model_is_overfitted(epoch,rmse_conv,over,opat)):
                final_epoch[:]= [epoch-1-opat,'overfit']
                return True
            elif(model_is_converged(epoch,rmse_conv,delta,cpat)):
                final_epoch[:] = [epoch-1-cpat,'converged']
                return True
            else:
                return False
    else:
        return False

def test(datasets,dfs,phys_model,model,target_norm):
    true_list,pred_list,phys_list,ml_list = [],[],[],[]
    pred_list = []
    phys_list = []
    ml_list = []
    rmse = []
    for i in range(len(datasets)):
        dataset = datasets[i]
        df = dfs[i]
        ml = np.array(model.predict(dataset)).flatten()
        if(target_norm):
            ml = target_norm.untransform(ml)
        phys = df[phys_model]
        true = df.expt
        pred = df[phys_model] + ml
        
        true_list.append(true)
        pred_list.append(pred)
        phys_list.append(phys)
        ml_list.append(ml)
        rmse.append(rmsd(true,pred))
    return true_list,pred_list,phys_list,ml_list,rmse

def get_max_checkpoints(hyper_params):
    if(using_early_stopping(hyper_params)):
        return 1 + max(hyper_params['es_over_patience'],hyper_params['es_conv_patience'])
    else:
        return 1

def get_graph_conv_checkpoint(epoch,model_folder):
    checkpoint_file = model_folder + 'checkpoints/checkpoint'
    with open(checkpoint_file, 'r') as file:
        lines = file.readlines()
    lines[0] = 'model_checkpoint_path: "ckpt-'+str(epoch)+'"\n'
    with open(checkpoint_file, 'w') as file:
        for line in lines:
            file.write(line)
    delete_graph_conv_checkpoints(model_folder+'checkpoints/','ckpt-'+str(epoch))

def train_model(datasets,dfs,phys_model,model_type,hyper_params,dataset_names,folder,iter,device,target_norm,print_results,with_target_noise):
    model_folder = folder + 'model'+str(iter)+'/'
    model = create_model(datasets,model_type,hyper_params,model_folder,device)
    dif = 0
    rmse_conv = [[] for _ in datasets]
    max_epochs = get_max_epochs(hyper_params,model_type)
    epoch = 1
    loss_conv = []
    final_epoch = [0,'']
    while(not should_stop_training(epoch,max_epochs,hyper_params,rmse_conv,final_epoch)):
        if(with_target_noise):
            y = datasets[0].y + 1.0*np.random.randn(*datasets[0].y.shape)*dfs[0].uncertainty
            dataset = dc.data.NumpyDataset(X=datasets[0].X,y = y)
            loss = model.fit(dataset, nb_epoch=1,max_checkpoints_to_keep = get_max_checkpoints(hyper_params), checkpoint_interval = 1000000)
        else:
            loss = model.fit(datasets[0], nb_epoch=1,max_checkpoints_to_keep = get_max_checkpoints(hyper_params), checkpoint_interval = 1000000)
        true,pred,phys,ml,rmse = test(datasets,dfs,phys_model,model,target_norm)
        loss_conv.append(loss)
        for r in range(len(rmse_conv)):
            rmse_conv[r].append(rmse[r])
        epoch += 1

    checkpoint_num = epoch - final_epoch[0]
    if(model_type == 'mpnn' or model_type == 'custom_mpnn' or model_type == 'dual_mpnn' or model_type == 'node_pred_mpnn' or model_type == 'pairwise_pred_mpnn'):
        delete_mpnn_checkpoints(model_folder+'checkpoints/',f'checkpoint{checkpoint_num}.pt')
        model.restore()#model_folder+f'checkpoints/checkpoint{checkpoint_num}.pt')
    elif(model_type == 'graphconv' and final_epoch[1] != 'max_epoch'):
        get_graph_conv_checkpoint(final_epoch[0],model_folder)
        model.restore()#model_folder+f'checkpoints/checkpoint{checkpoint_num}.pt')
    true,pred,phys,ml,rmse = test(datasets,dfs,phys_model,model,target_norm)
    
    if(print_results):
        print_model(iter,final_epoch,loss,dataset_names,rmse)
    save_model_data(model_folder,true,pred,phys,ml,iter,rmse_conv,loss_conv,final_epoch,dataset_names)
    return rmse,final_epoch#epoch-1

def run_experiments(Xs, dfs, phys_models,model_type,hyper_params,iter,dataset_names,folder,folder_list,normalize_targets=False,norm_datasets=[],device = 'cuda',print_results = 1,with_target_noise=False,include_uncertainty=False):
    folder_list.append(folder)
    for phys_model in phys_models:
        phys_model_folder = folder + phys_model + '/'
        if(print_results):
            print(get_physics_model_name(phys_model))
            
        datasets = []
        norm_y = []
        ys = []
        for i in range(len(dfs)):
            if(include_uncertainty):
                y = np.zeros((len(dfs[i]),2))
                y[:,0] = dfs[i].expt - dfs[i][phys_model]
                if(dataset_names[i] != 'amino'):
                    y[:,1] = dfs[i].uncertainty
            else:
                y = dfs[i].expt - dfs[i][phys_model]
            ys.append(y)
            if(dataset_names[i] in norm_datasets):
                norm_y += list(y)

        target_norm=None
        # print(ys[2].mean(),ys[2].std())
        if(normalize_targets):
            target_norm = ZScoreNormalizer()
            target_norm.fit(norm_y)
            for i in range(len(ys)):
                ys[i] = target_norm.transform(ys[i])
        # print(ys[2].mean(),ys[2].std())
        for i in range(len(dfs)):
            dataset = dc.data.NumpyDataset(X=Xs[i],y=ys[i])
            datasets.append(dataset)
            
        rmses = [[] for _ in range(len(dfs))]
        epochs = []
        conv_reasons = []
        for i in range(iter):
            rmse,final_epoch = train_model(datasets,dfs,phys_model,model_type,hyper_params,dataset_names,phys_model_folder,i,device,target_norm,print_results > 1,with_target_noise)
            for r in range(len(rmses)):
                rmses[r].append(rmse[r])
            epochs.append(final_epoch[0])
            conv_reasons.append(final_epoch[1])
        save_combined_data(phys_model_folder,rmses,epochs,conv_reasons,dataset_names)
        # save_params(hyper_params,folder)
        torch.cuda.empty_cache()
        if(print_results):
            print_summary_data(phys_model_folder)
            print()
    return 

def get_featurizer(name,feat_params):
    if(name == 'MolGraphConvFeaturizer'):
        return dc.feat.MolGraphConvFeaturizer(use_edges=True,use_chirality=feat_params['use_chirality'],use_partial_charge=feat_params['use_partial_charge'])
    elif(name == 'ConvMolFeaturizer'):
        return dc.feat.ConvMolFeaturizer(per_atom_fragmentation=False)
    elif(name == 'MPNN_custom'):
        print('not yet defined')
    elif(name == 'GraphConv_custom'):
        print('not yet defined')
        #featurizer here which takes feat_params
        #this includes list of features for node and edge
        #and how edges are decided
    else:
        print('enter a valid featurizer name')
    return 0
    
#are below necessary??
# def run_feat_param_combos(dfs,dataset_names,feat_param_combos,hyper_combos,folder_base,featurizer_name,input_file_paths = None,train_val_df = None):
#     for params in feat_param_combos:
#         feat_params = dict(zip(feat_params_dict.keys(), params))
#         feat_folder = create_folder_name(folder_base,feat_params)
    
#         featurizer = get_featurizer(featurizer_name,feat_param_combos)
#         Xs = []
#         for i in range(len(dfs)):
#             df = dfs[i]
#             if((featurizer_name == 'MolGraphConvFeaturizer') or (featurizer_name == 'ConvMolFeaturizer')):
#                 X = featurizer.featurize(df.smiles)
#             else:
#                 X = featurizer.featurize(df,input_file_paths[i])
#             Xs.append(X)

# def run_hyper_combos():



