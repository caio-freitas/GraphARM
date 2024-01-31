
from tqdm import tqdm
import torch
import wandb
import torch.nn as nn
import logging

from benchmarks.GraphARM.models import DiffusionOrderingNetwork, DenoisingNetwork
from benchmarks.GraphARM.utils import NodeMasking

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class GraphARM(nn.Module):
    '''
    Class to encapsule DiffusionOrderingNetwork and DenoisingNetwork, as well as the training loop
    for both with diffusion and denoising steps.
    '''
    def __init__(self,
                 dataset,
                 denoising_network,
                 diffusion_ordering_network,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(GraphARM, self).__init__()
        self.device = device
        self.diffusion_ordering_network = diffusion_ordering_network
        self.diffusion_ordering_network.to(device)
        self.denoising_network = denoising_network
        self.denoising_network.to(device)
        self.masker = NodeMasking(dataset)


        self.denoising_optimizer = torch.optim.Adam(self.denoising_network.parameters(), lr=1e-3, betas=(0.9, 0.999))
        self.ordering_optimizer = torch.optim.Adam(self.diffusion_ordering_network.parameters(), lr=5e-3, betas=(0.9, 0.999))


    def node_decay_ordering(self, datapoint):
        '''
        Returns node order for a given graph, using the diffusion ordering network.
        '''
        p = datapoint.clone().to(self.device)
        node_order = []
        sigma_t_dist_list = []
        
        for i in range(p.x.shape[0]):
            # use diffusion ordering network to get probabilities
            sigma_t_dist = self.diffusion_ordering_network(p, node_order)
            # sample (only unmasked nodes) from categorical distribution to get node to mask
            unmasked = ~self.masker.is_masked(p)
            sigma_t_dist_list.append(sigma_t_dist.flatten())
            sigma_t = torch.distributions.Categorical(probs=sigma_t_dist[unmasked].flatten()).sample()

            # get node index
            sigma_t = torch.where(unmasked.flatten())[0][sigma_t.long()]
            node_order.append(sigma_t)
            # mask node
            p = self.masker.mask_node(p, sigma_t)
        return node_order, sigma_t_dist_list

    def uniform_node_decay_ordering(self, datapoint):
        '''
        Samples next node from uniform distribution 
        '''
        p = datapoint.clone()
        return torch.randperm(p.x.shape[0]).tolist()


    def generate_diffusion_trajectories(self, graph, M):
        '''
        Generates M diffusion trajectories for a given graph,
        using the node decay ordering mechanism.
        '''
        original_data = graph.clone().to(self.device)
        diffusion_trajectories = []
        for m in range(M):
            node_order, sigma_t_dist = self.node_decay_ordering(graph)
            node_order_invariate = node_order

            # create diffusion trajectory
            diffusion_trajectory = [original_data]
            masked_data = graph.clone()
            for i in range(len(node_order)):
                node = node_order[i]
                masked_data = masked_data.clone().to(self.device)
                
                masked_data = self.masker.mask_node(masked_data, node)
                diffusion_trajectory.append(masked_data)
                # don't remove last node
                if i < len(node_order)-1:
                    masked_data = self.masker.remove_node(masked_data, node)
                    node_order = [n-1 if n > node else n for n in node_order] # update node order to account for removed node

            diffusion_trajectories.append([diffusion_trajectory, node_order_invariate, sigma_t_dist])
        return diffusion_trajectories

    def preprocess(self, graph):
        '''
        Preprocesses graph to be used by the denoising network.
        '''
        graph = graph.clone()
        # node and edge types to idx
        graph = self.masker.idxify(graph)
        graph = self.masker.fully_connect(graph)
        return graph

    def train_step(
            self,
            train_data,
            val_data,
            M = 4, # number of diffusion trajectories to be created for each graph
        ):
        
        self.denoising_optimizer.zero_grad()
        self.ordering_optimizer.zero_grad()


        self.denoising_network.train()
        self.diffusion_ordering_network.eval()
        loss = torch.tensor(0.0, requires_grad=True)
        acc_loss = 0.0
        with tqdm(train_data) as pbar:
            for graph in pbar:
                # preprocess graph
                graph = self.preprocess(graph)
                diffusion_trajectories = self.generate_diffusion_trajectories(graph, M)  
                # predictions & loss
                for diffusion_trajectory, node_order, sigma_t_dist in diffusion_trajectories:
                    G_0 = diffusion_trajectory[0]

                    for t in range(len(node_order)):
                        for k in range(len(node_order) - t - 1):# until t
                            G_pred = diffusion_trajectory[t+1].clone()                              

                            # predict node and edge type distributions
                            node_type_probs, edge_type_probs = self.denoising_network(G_pred.x, G_pred.edge_index, G_pred.edge_attr)

                            w_k = sigma_t_dist[t][node_order[k]]
                            w_k = w_k.detach()
                            wandb.log({"target_node_ordering_prob": w_k.item()})
                            # calculate loss
                            loss = self.vlb(G_0, node_type_probs, edge_type_probs, w_k, node_order[k], node_order, t, M) # cumulative, to join (k) from all previously denoised nodes
                            wandb.log({"vlb": loss.item()})

                            acc_loss += loss.item()
                            # backprop (accumulated gradients)
                            loss.backward()
                            pbar.set_description(f"Loss: {acc_loss:.4f}")
        
        # update parameters using accumulated gradients
        self.denoising_optimizer.step()
        
        # log loss
        wandb.log({"loss": acc_loss})

        self.denoising_optimizer.zero_grad()
        self.ordering_optimizer.zero_grad()
        
        ## validation batch (optimizing diffusion ordering network)
        self.denoising_network.eval()
        self.diffusion_ordering_network.train()

        reward = torch.tensor(0.0, requires_grad=True)
        acc_reward = 0.0
        with tqdm(val_data) as pbar:
            for graph in pbar:
                # preprocess graph
                graph = self.preprocess(graph)
                diffusion_trajectories = self.generate_diffusion_trajectories(graph, M)  
                # predictions & loss
                for diffusion_trajectory, node_order, sigma_t_dist_unmasked  in diffusion_trajectories:
                    G_0 = diffusion_trajectory[0]
                    for t in range(len(node_order)):
                        for k in range(len(node_order) - t - 1):
                            G_pred = diffusion_trajectory[t+1].clone()                              

                            # predict node and edge type distributions
                            node_type_probs, edge_type_probs = self.denoising_network(G_pred.x, G_pred.edge_index, G_pred.edge_attr)
                            # not require grad for output of denoising network
                            node_type_probs = node_type_probs.detach()
                            edge_type_probs = edge_type_probs.detach()

                            w_k = sigma_t_dist_unmasked[t][node_order[k]]
                            wandb.log({"target_node_ordering_prob": w_k.item()})
                            # calculate loss
                            reward = self.vlb(G_0, node_type_probs, edge_type_probs, w_k, node_order[k], node_order, t, M)
                            wandb.log({"vlb": reward.item()})
                            acc_reward -= reward.item()
                            # backprop (accumulated gradients)
                            reward.backward()
                            pbar.set_description(f"Reward: {acc_reward:.4f}")

        
        wandb.log({"reward": acc_reward})
        # update parameters (REINFORCE algorithm)
        self.ordering_optimizer.step()
        


    def vlb(self, G_0, node_type_probs, edge_type_probs, w_k, node, node_order, t, M):
        '''
        Calculates the variational lower bound (VLB) for a given node and edge type distribution, 
        relative to the original graph G_0.
        ** Ignores the KL divergence term, as described in the paper.
        ** edge_type_probs only contains probabilities for edges between node and node_order[t:] - see generate_diffusion_trajectories()
        '''
        T = len(node_order)
        n_i = G_0.x.shape[0]
        # retrieve edge type from G_t.edge_attr, edges between node and node_order[t:]
        edge_attrs_matrix = G_0.edge_attr.reshape(T, T)
        original_edge_types = torch.index_select(edge_attrs_matrix[node], 0, torch.tensor(node_order[t:]).to(self.device))
        # calculate probability of edge type
        p_edges = torch.gather(edge_type_probs, 1, original_edge_types.reshape(-1, 1))
        log_p_edges = torch.sum(torch.log(p_edges))
        # log_p_edges = torch.sum(torch.tensor([0]))
        wandb.log({"target_node_type_prob": node_type_probs[G_0.x[node]].item()})
        wandb.log({"target_edges_log_prob": log_p_edges})
        # calculate loss
        log_p_O_v =  log_p_edges + torch.log(node_type_probs[G_0.x[node]])
        loss = -(n_i/T)*log_p_O_v*w_k/M # cumulative, to join (k) from all previously denoised nodes
        return loss


    def predict_new_node(self, 
                         graph, 
                         sampling_method="argmax",
                         preprocess=True):
        '''
        Predicts the value of a new node for graph as well as it's connection to all previously denoised nodes.
        sampling_method: "argmax" or "sample"
        - argmax: select node and edge type with highest probability
        - sample: sample node and edge type from multinomial distribution
        '''
        assert sampling_method in ["argmax", "sample"], "sampling_method must be either 'argmax' or 'sample'"
        with torch.no_grad():
            if preprocess:
                graph = self.preprocess(graph)
            # add masked node to graph
            graph = self.masker.add_masked_node(graph)
            # predict node type
            node_type_probs, edge_type_probs = self.denoising_network(graph.x, graph.edge_index, graph.edge_attr)
            
            # sample node type
            if sampling_method == "sample":
                node_type = torch.distributions.Categorical(probs=node_type_probs.squeeze()).sample()
            elif sampling_method == "argmax":
                node_type = torch.argmax(node_type_probs.squeeze(), dim=-1)

            # sample edge type
            if sampling_method == "sample":
                new_connections = torch.multinomial(edge_type_probs.squeeze(), num_samples=1, replacement=True)
            elif sampling_method == "argmax":
                new_connections = torch.argmax(edge_type_probs.squeeze(), dim=-1)
            # no need to filter connection to previously denoised nodes, assuming only one new node is added at a time

            # lookup dictionary to get node type
            node_type = torch.tensor(next(key for key, value in self.masker.node_type_to_idx.items() if value == node_type.item()), dtype=torch.int32).to(self.device)

            # lookup dictionary to get edge type
            if new_connections.any():
                new_connections = torch.tensor([list(self.masker.edge_type_to_idx.keys())[list(self.masker.edge_type_to_idx.values()).index(edge_idx)] for edge_idx in new_connections]).to(self.device)

        return node_type, new_connections


    def save_model(self, denoising_network_path="denoising_network.pt", diffusion_ordering_network_path="diffusion_ordering_network.pt"):
        torch.save(self.denoising_network.state_dict(), denoising_network_path)
        torch.save(self.diffusion_ordering_network.state_dict(), diffusion_ordering_network_path)

    def load_model(self,
                   denoising_network_path="denoising_network_overfit.pt",
                   diffusion_ordering_network_path="diffusion_ordering_network_overfit.pt"):
        self.denoising_network.load_state_dict(torch.load(denoising_network_path, map_location=self.device ))
        self.diffusion_ordering_network.load_state_dict(torch.load(diffusion_ordering_network_path, map_location=self.device))