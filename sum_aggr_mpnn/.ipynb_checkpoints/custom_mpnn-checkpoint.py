import torch_geometric.transforms as T
from torch_geometric.nn import NNConv, global_max_pool, MLP
from torch_geometric.data import Data, DataLoader
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d, Dropout
import torch.nn.functional as F
from dgl.nn.pytorch import Set2Set
from dgllife.model.gnn.mpnn import MPNNGNN
import torch
import deepchem as dc
from deepchem.models.optimizers import Optimizer, LearningRateSchedule
import torch.nn as nn
from typing import Dict, Union, Optional
class Adam(Optimizer):
    def __init__(self,
                 learning_rate: Union[float, LearningRateSchedule] = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-08,
                 weight_decay: float = 0):
        super(Adam, self).__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

    def _create_pytorch_optimizer(self, params):
        import torch
        if isinstance(self.learning_rate, LearningRateSchedule):
            lr = self.learning_rate.initial_rate
        else:
            lr = self.learning_rate
        return torch.optim.Adam(params,
                                lr=lr,
                                betas=(self.beta1, self.beta2),
                                eps=self.epsilon,
                                weight_decay=self.weight_decay)

class CustomMPNNPredictor(torch.nn.Module):
    def __init__(self,
                 num_node_features, 
                 num_edge_features,
                 node_out_feats=64,
                 edge_hidden_feats=128,
                 n_tasks=1,
                 num_step_message_passing=3,
                 num_step_set2set=6,
                 num_layer_set2set=3,
                 nfeat_name: str = 'x',
                 efeat_name: str = 'edge_attr'):
        super(CustomMPNNPredictor,self).__init__()
        self.nfeat_name = nfeat_name
        self.efeat_name = efeat_name
        self.gnn = MPNNGNN(node_in_feats=num_node_features,
                           node_out_feats=node_out_feats,
                           edge_in_feats=num_edge_features,
                           edge_hidden_feats=edge_hidden_feats,
                           num_step_message_passing=num_step_message_passing)
        self.readout = Set2Set(input_dim=node_out_feats,
                               n_iters=num_step_set2set,
                               n_layers=num_layer_set2set)
        self.predict = nn.Sequential(
            nn.Linear(node_out_feats, node_out_feats),
            nn.ReLU(),
            nn.Linear(node_out_feats, n_tasks)
        )
        
    def forward(self, g):
        node_feats = g.ndata[self.nfeat_name]
        edge_feats = g.edata[self.efeat_name]
        
        node_feats = self.gnn(g, node_feats, edge_feats)

        num_nodes = torch.cumsum(g.batch_num_nodes(),dim=0)
        num_nodes = torch.cat((torch.tensor([0]).to(torch.device('cuda')), num_nodes))
        
        graph_feats = [node_feats[num_nodes[i]:num_nodes[i+1]].sum(dim=0) for i in range(len(num_nodes) - 1)]
        graph_feats = torch.stack(graph_feats)
        # graph_feats = self.readout(g, node_feats)
        return self.predict(graph_feats)

    
        
class CustomMPNN(dc.models.TorchModel):
    def __init__(self,
                 n_tasks: int,
                 node_out_feats: int = 64,
                 edge_hidden_feats: int = 128,
                 num_step_message_passing: int = 3,
                 num_step_set2set: int = 6,
                 num_layer_set2set: int = 3,
                 mode: str = 'regression',
                 number_atom_features: int = 30,
                 number_bond_features: int = 11,
                 n_classes: int = 2,
                 self_loop: bool = False,
                 weight_decay: float = 0.0,
                 **kwargs):
        model = CustomMPNNPredictor(n_tasks=n_tasks,
                     node_out_feats=node_out_feats,
                     edge_hidden_feats=edge_hidden_feats,
                     num_step_message_passing=num_step_message_passing,
                     num_step_set2set=num_step_set2set,
                     num_layer_set2set=num_layer_set2set,
                     num_node_features=number_atom_features,
                     num_edge_features=number_bond_features)
        loss = dc.models.losses.L2Loss()
        output_types = ['prediction']
        super(CustomMPNN, self).__init__(model, loss, 
                                           output_types=output_types,
                                           optimizer=Adam(learning_rate=0.001,weight_decay=weight_decay), 
                                           **kwargs)
        self._self_loop = self_loop
    
    def _prepare_batch(self,batch):
        try:
            import dgl
        except:
            raise ImportError('This class requires dgl.')

        inputs, labels, weights = batch
        dgl_graphs = [
            graph.to_dgl_graph(self_loop=self._self_loop) for graph in inputs[0]
        ]
        inputs = dgl.batch(dgl_graphs).to(self.device)
        _, labels, weights = super(CustomMPNN, self)._prepare_batch(
            ([], labels, weights))
        return inputs, labels, weights



