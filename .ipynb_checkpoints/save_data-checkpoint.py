import deepchem as dc
import os
import sys
os.environ['DGLBACKEND'] = 'pytorch'
import numpy as np
import pandas as pd
import dgl
import torch
import pickle
from deepchem.feat.graph_data import GraphData
from deepchem.models import MPNNModel
from deepchem.feat import MolGraphConvFeaturizer

import math

import copy
from utils import *
from itertools import product

def save_model_data(folder,true,pred,phys,ml,iter,rmse_conv,loss_conv,final_epoch,dataset_names):
    for i in range(len(dataset_names)):
        df = pd.DataFrame()
        df['expt'] = true[i]
        df['physics_ml'] = pred[i]
        df['physics'] = phys[i]
        df['ml'] = ml[i]
        if os.path.isdir(folder):
            df.to_csv(folder+dataset_names[i]+'.csv',index=False)
        else:
            os.makedirs(folder)
            df.to_csv(folder+dataset_names[i]+'.csv',index=False)
    conv_df = pd.DataFrame()
    conv_df['epoch'] = np.arange(1,len(rmse_conv[0])+1)
    conv_df['loss'] = loss_conv
    for i in range(len(dataset_names)):
        conv_df[dataset_names[i]] = rmse_conv[i]
    conv_df['final_model'] = ''
    conv_df.at[final_epoch[0]-1,'final_model'] = final_epoch[1]
    conv_df['rel_val_overfitting'] = (conv_df[dataset_names[1]] - conv_df[dataset_names[0]])/(conv_df[dataset_names[1]])
    conv_df.to_csv(folder+'convergence.csv', index=False)

def save_combined_data(folder,rmses,epochs,conv_reasons,dataset_names):
    df = pd.DataFrame()
    df['model_num'] = np.arange(0,len(rmses[0]))
    df['epochs'] = epochs
    for i in range(len(dataset_names)):
        df[dataset_names[i]] = rmses[i]
    df['conv_criterion'] = conv_reasons
    df.to_csv(folder+'rmse.csv', index=False)

def save_datasets(folder,Xs,dfs,dataset_names,featurizer,normalizer=None):
    dict = {
        'Xs':Xs,
        'dfs':dfs,
        'dataset_names':dataset_names,
        'featurizer':featurizer,
        'normalizer':normalizer
    }
    if not os.path.isdir(folder):
        os.makedirs(folder)
    file_name = folder+'datasets.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(dict, file)

def load_saved_datasets(file_name):
    with open(file_name, 'rb') as file:
        dict = pickle.load(file)

    return dict['Xs'],dict['dfs'],dict['dataset_names'],dict['featurizer'],dict['normalizer']

def save_params(params,folder,file_name='params'):
    filepath = folder + file_name+'.json'
    with open(filepath, "w") as outfile: 
        json.dump(params, outfile)

def generate_summary_dfs(folders,phys_models,num_datasets = 4):
    cols = ['settings']#,'TIP3P','CHA-GB','ZAP9','mbondi','IGB5','AASC','ML alone']
    for phys in phys_models:
        cols.append(get_physics_model_name(phys))
    epoch_df = pd.DataFrame(columns = cols)
    
    # print(epochs_df)
    ens_dfs = [pd.DataFrame(columns = cols) for i in range(num_datasets)]
    mean_dfs = [pd.DataFrame(columns = cols) for i in range(num_datasets)]
    done_phys = 0
    len_datasets = []
    for folder in folders:
        phys_dirs = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)) and not d.startswith('.')]
        mean_rows = [[folder.split('/', 1)[1]] for i in range(num_datasets)]
        ens_rows = [[folder.split('/', 1)[1]] for i in range(num_datasets)]
        epoch_row = [folder.split('/', 1)[1]]
        phys_rows = [['physics model'] for i in range(num_datasets)]
        
        for dir in phys_dirs:
            phys_folder = folder+dir+'/'       
            model_dirs = [d for d in os.listdir(phys_folder) if os.path.isdir(os.path.join(phys_folder, d)) and not d.startswith('.')]
            dataset_names = [d for d in os.listdir(phys_folder+model_dirs[0]+'/') if os.path.isfile(os.path.join(phys_folder+model_dirs[0]+'/', d)) and d.endswith('.csv') and not d.startswith('convergence')]
            rmse_df = pd.read_csv(phys_folder+'rmse.csv')
            # print(dataset_names)

            
            for i in range(len(dataset_names)):
                dataset = dataset_names[i]
                dataset = dataset[:-4] #removes ".csv"
                ml_mean=(np.mean(rmse_df[dataset]))
                ml_std=(np.std(rmse_df[dataset]))           
                mean_rows[i].append(f'{ml_mean:.2f} ± {ml_std:.2f}')
                # print(mean_rows)
                if(not done_phys):
                    df = pd.read_csv(phys_folder+model_dirs[0]+'/'+dataset+'.csv')
                    len_datasets.append(len(df))
                    phys_rows[i].append(f'{rmsd(df.expt,df.physics):.2f}')
                    
            for i in range(len(dataset_names)):
                array = np.zeros((len_datasets[i],len(model_dirs)))
                for j in range(len(model_dirs)):
                    df = pd.read_csv(phys_folder+model_dirs[j]+'/'+dataset_names[i])
                    array[:,j] = df.physics_ml
                mean_array = np.mean(array,axis=1)
                ens_rows[i].append(f'{rmsd(df.expt,mean_array):.2f}')
            epoch_row.append(f'{np.mean(rmse_df.epochs):.2f} ± {np.std(rmse_df.epochs):.2f}')
        epoch_df.loc[len(epoch_df.index)] = epoch_row
        for i in range(len(mean_dfs)):
            # print(len(mean_rows[i]))
            if(not done_phys):
                mean_dfs[i].loc[len(mean_dfs[i].index)] = phys_rows[i]
                ens_dfs[i].loc[len(ens_dfs[i].index)] = phys_rows[i]
                
            mean_dfs[i].loc[len(mean_dfs[i].index)] = mean_rows[i]
            ens_dfs[i].loc[len(ens_dfs[i].index)] = ens_rows[i]
        done_phys = 1
        
    return ens_dfs,mean_dfs,epoch_df,[d[:-4] for d in dataset_names]
            
def save_summary_dfs(ens_dfs,mean_dfs,epoch_df,dataset_names,folder):
    folder_base = folder+'summary_tables/'
    if not os.path.isdir(folder_base):
        os.makedirs(folder_base)
    epoch_df.to_csv(folder_base+'summary_epochs.csv',index=False)
    for i in range(len(dataset_names)):
        ens_dfs[i].to_csv(folder_base+'summary_ens_'+dataset_names[i]+'.csv',index=False)
        mean_dfs[i].to_csv(folder_base+'summary_means_'+dataset_names[i]+'.csv',index=False)
    







    
