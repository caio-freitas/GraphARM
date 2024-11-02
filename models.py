import torch
from torch import nn
from torch_geometric.nn.conv import RGCNConv
from torch_geometric.nn import GAT
from torch_geometric.utils import add_self_loops, degree
from torch.nn import functional as F
from torch.nn import Linear, ReLU
import math
from torch_geometric.nn import MessagePassing


class RGCN(nn.Module):
    def __init__(self, num_relations, hidden_dim, out_channels=1, num_layers=3, device='cpu'):
        super(RGCN, self).__init__()
        self.device = device
        self.embedding_dim = hidden_dim
        self.num_layers = num_layers

        self.conv = []

        # Define R-GCN layers
        for layer in range(num_layers - 1):
            self.conv.append(RGCNConv(in_channels=hidden_dim, out_channels=hidden_dim, num_relations=num_relations, num_bases=2).to(self.device))
        
        self.conv.append(RGCNConv(in_channels=hidden_dim, out_channels=out_channels, num_relations=num_relations, num_bases=2).to(self.device))
    
    def forward(self, x, edge_index, edge_type):
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_type = edge_type.to(self.device)
        
        # R-GCN layers
        for layer in range(self.num_layers):
            x = self.conv[layer](x, edge_index, edge_type)
        return x

class DiffusionOrderingNetwork(nn.Module):
    '''
    at each diffusion step t, we sample from this network to select a node 
    v_sigma(t) to be absorbed and obtain the corresponding masked graph Gt
    '''
    def __init__(self,
                 node_feature_dim,
                 num_node_types,
                 num_edge_types,
                 num_layers=3,
                 out_channels=1,
                 hidden_dim=32,
                 num_heads=6,
                 device='cpu'):
        super(DiffusionOrderingNetwork, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels

        num_node_types += 1 # add one for masked node type
        num_edge_types += 2 # add one for masked edge type and one for empty edge type
        

        # add positional encodings into node features
        self.embedding = nn.Embedding(num_embeddings=num_node_types, embedding_dim=hidden_dim).to(self.device)

        # self.gat = GAT(
        #     in_channels=hidden_dim,
        #     out_channels=out_channels,
        #     hidden_channels=hidden_dim * num_heads,
        #     num_layers=num_layers,
        #     dropout=0,
        #     heads=hidden_dim,
        #     residual=True
        # )
        
        # Create an instance of the RGCN model
        self.gat = RGCN(num_relations=num_edge_types,
                        hidden_dim=self.hidden_dim,
                        out_channels=self.out_channels,
                        num_layers=num_layers,
                        device=device).to(self.device)
        
        # initialize positional encodings
        MAX_NODES = 10000
        self.pe = self.positionalencoding(MAX_NODES).to(self.device)


    def positionalencoding(self, lengths):
        '''
        From Chen, et al. 2021 (Order Matters: Probabilistic Modeling of Node Sequences for Graph Generation)
        * lengths: length(s) of graph in the batch
        '''
        l_t = lengths # .max() # use when parallelizing
        pes = torch.zeros([l_t, self.out_channels], device=self.device)
        position = torch.arange(0, l_t, device=self.device).unsqueeze(1) + 1
        div_term = torch.exp((torch.arange(0, self.out_channels, 2, dtype=torch.float, device=self.device) *
                              -(math.log(10000.0) / self.out_channels)))
        pes[:,0::2] = torch.sin(position.float() * div_term)
        pes[:,1::2] = torch.cos(position.float() * div_term)
        return pes

    def forward(self, G, node_order=None):
        '''
        node_order: list of absorbed nodes so far
        '''
        # list of not absorbed nodes (G.x.shape[0], except for nodes in node_order)
        unmasked = torch.tensor([node for node in range(G.x.shape[0]) if node not in node_order], device=self.device)

        h = self.embedding(G.x.squeeze().long().to(self.device))

        # # Positional encoding
        for t in range(len(node_order)):
            h[node_order[t], :] += self.pe[t, :].to(self.device)
        h = self.gat(h, G.edge_index.long().to(self.device), G.edge_attr.long().to(self.device))

        if unmasked.numel() > 0:
            h_unmasked = h[unmasked, :]
            # softmax: h over h_not_absorbed
            # make sure values are positive and sum to 1 (for unmasked nodes)
            h = torch.exp(h) / torch.sum(torch.exp(h_unmasked), dim=0)
            h[node_order, :] *= 0 # zero the probability for already absorbed nodes
        else:
            h = torch.exp(h) / torch.sum(torch.exp(h), dim=0)
        return h  # outputs probabilities for a categorical distribution over nodes
    
    
class MPLayer(MessagePassing):
    '''
    Custom message passing layer for the GraphARM model
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='sum') #  "Max" aggregation.
        self.f = nn.Sequential(Linear(3 * in_channels, out_channels),
                       nn.ReLU(),
                       Linear(out_channels, out_channels)) # MLP for message construction
        self.g = nn.Sequential(Linear(3 * in_channels, out_channels),
                          nn.ReLU(),
                          Linear(out_channels, out_channels)) # MLP for attention coefficients
        
        self.gru = nn.GRU(2*out_channels, out_channels)
        
    def forward(self, x, edge_index, edge_attr):
        '''
        x has shape [N, in_channels]
        edge_index has shape [2, E]
        **self-loops should be added in the preprocessing step (fully connecting the graph)
        '''

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out, _ = self.gru(torch.cat([x, out], dim=-1)) # discard final hidden state
        return out

    def message(self, x_i, x_j, edge_attr):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        h_vi = x_i
        h_vj = x_j
        h_eij = edge_attr

        m_ij = self.f(torch.cat([h_vi, h_vj, h_eij], dim=-1))
        a_ij = self.g(torch.cat([h_vi, h_vj, h_eij], dim=-1))
        return m_ij * a_ij


class DenoisingNetwork(nn.Module):
    def __init__(self,
                node_feature_dim,
                edge_feature_dim,
                num_node_types,
                num_edge_types,
                num_layers=5,
                hidden_dim=256,
                K=20,
                device='cpu'):
        super().__init__()
        self.device = device
        num_edge_types += 1 # add one for empty edge type
        self.K = K
        self.num_layers = num_layers
        self.node_embedding = Linear(node_feature_dim, hidden_dim).to(self.device)
        self.edge_embedding = Linear(edge_feature_dim, hidden_dim).to(self.device)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(MPLayer(hidden_dim, hidden_dim)).to(self.device)

        self.mlp_alpha = nn.Sequential(Linear(3*hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       Linear(hidden_dim, self.K)).to(self.device)
        
        self.node_pred_layer = nn.Sequential(Linear(2*hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       Linear(hidden_dim, num_node_types)).to(self.device)
        
        self.edge_pred_layer = nn.Sequential(Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       Linear(hidden_dim, num_edge_types*K)).to(self.device)


        
    def forward(self, x, edge_index, edge_attr, v_t=None):
        # make sure x and edge_attr are of type float, for the MLPs
        x = x.float().to(self.device)
        edge_attr = edge_attr.float().to(self.device)

        h_v = self.node_embedding(x)
        h_e = self.edge_embedding(edge_attr.reshape(-1, 1))
        
        for l in range(self.num_layers):
            h_v = self.layers[l](h_v, edge_index, h_e)


        # graph-level embedding, from average pooling layer
        graph_embedding = torch.mean(h_v, dim=0)

        # repeat graph embedding to have the same shape as h_v
        graph_embedding = graph_embedding.repeat(h_v.shape[0], 1)

        node_pred = self.node_pred_layer(torch.cat([graph_embedding, h_v], dim=1)) # hidden_dim + 1
        
        # edge prediction follows a mixture of multinomial distribution, with
        # the Softmax(sum(mlp_alpha([graph_embedding, h_vi, h_vj])))
        alphas = torch.zeros(h_v.shape[0], self.K)
        if v_t is None:
            v_t = h_v.shape[0] - 1# node being masked, this assumes that the masked node is the last node in the graph
        h_v_t = h_v[v_t, :].repeat(h_v.shape[0], 1)

        alphas = self.mlp_alpha(torch.cat([graph_embedding, h_v_t, h_v], dim=1))

        alphas = F.softmax(torch.sum(alphas, dim=0, keepdim=True), dim=1)

        p_v = F.softmax(node_pred, dim=-1)
        log_theta = self.edge_pred_layer(h_v)
        log_theta = log_theta.view(h_v.shape[0], -1, self.K) # h_v.shape[0] is the number of steps (nodes) (block size)
        p_e = torch.sum(alphas * F.softmax(log_theta, dim=1), dim=-1) # softmax over edge types

        p_v = p_v.to(self.device) 
        p_e = p_e.to(self.device) 

        return p_v, p_e