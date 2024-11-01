
from tqdm import tqdm
import torch
import wandb
import torch.nn as nn
import logging

from .models import DiffusionOrderingNetwork, DenoisingNetwork
from .utils import NodeMasking

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


        self.denoising_optimizer = torch.optim.Adam(self.denoising_network.parameters(), lr=1e-4, betas=(0.9, 0.999))
        self.ordering_optimizer = torch.optim.Adam(self.diffusion_ordering_network.parameters(), lr=5e-4, betas=(0.9, 0.999))


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
            unmasked = torch.tensor([i not in node_order for i in range(p.x.shape[0])]).to(self.device)

            sigma_t_dist_list.append(sigma_t_dist.flatten())
            sigma_t = torch.distributions.Categorical(probs=sigma_t_dist[unmasked].flatten()).sample()

            # get node index
            sigma_t = torch.where(unmasked.flatten())[0][sigma_t.long()]
            node_order.append(sigma_t)
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
        graph = self.masker.idxify(graph)
        graph = self.masker.fully_connect(graph)
        return graph

    def compute_denoising_loss(self, diffusion_trajectory, node_order_invariate, sigma_t_dist_list):
        '''
        Computes the loss for the denoising network based on negative log-likelihood (NLL).
        '''
        loss = 0
        T = len(diffusion_trajectory) - 1  # Total number of time steps
        sigma_t = torch.stack(sigma_t_dist_list, dim=0)
        G_0 = diffusion_trajectory[0]  # Original graph
        for t in range(0, T):  # Start at 1 because 0 is the original graph
            graph_t = diffusion_trajectory[t]  # G_t
            graph_t_next = diffusion_trajectory[t+1]  # G_{t+1}
            
            # Predict node and edge types
            node_type_probs, edge_type_probs = self.denoising_network(graph_t_next.x, graph_t_next.edge_index, graph_t_next.edge_attr)

            # Compute NLL for node type
            # compute for all nodes, weight them by the sigma_t_dist at the original node order
            sigma_t_dist = sigma_t[t]
            sigma_t_dist = sigma_t_dist[sigma_t_dist != 0]
            
            # select elements from sigma_t_dist that correspond to the original node order
            node_probs = node_type_probs * sigma_t_dist.view(-1, 1).clone()
            # get probability of choosing correct node type
            correct_node_type = G_0.x[node_order_invariate[t]]
            
            nll_node = -torch.log(node_probs[:, correct_node_type].sum() + 1e-8)
            
            # Compute NLL for edge type
            edge_probs = edge_type_probs.view(-1, edge_type_probs.shape[-1])
            # get original edge index for each node being unmasked
            
            # get probability of choosing edge type for each edge
            # composing edge_type_probs with sigma_t_dist
            # P(edges) = P(node) * P(edge type|node)
            edge_probs = edge_probs * sigma_t_dist.view(-1, 1).clone()
            
            # get original edge type for each edge in G_0
            
            correct_edge_type = G_0.edge_attr[(G_0.edge_index[0] == node_order_invariate[t]) & (torch.tensor([G_0.edge_index[1][i] in node_order_invariate[t:] for  i in range(G_0.edge_index.shape[1])]))]
            # get probability of choosing correct edge type
            edge_probs = torch.gather(edge_probs, 1, correct_edge_type.view(-1, 1))
            nll_edge = -torch.log(edge_probs + 1e-8).sum()
            
            
            loss += nll_node.mean() + nll_edge.mean()

        return loss / T

    def compute_ordering_loss(self, diffusion_trajectories, M):
        '''
        Computes the loss for the diffusion ordering network using the REINFORCE algorithm.
        '''
        ordering_loss = 0
        for trajectory, node_order, sigma_t_dist_list in diffusion_trajectories:
            # Compute the reward as the negative denoising loss
            reward = -self.compute_denoising_loss(trajectory, node_order, sigma_t_dist_list)
            wandb.log({"reward": reward.item()})
            # REINFORCE update (policy gradient)
            # Calculate probability of trajectory using sigma_t_dist_list
            log_prob = torch.tensor(0.0, device=self.device)
            for t in range(len(sigma_t_dist_list)):
                log_prob = log_prob + torch.log(sigma_t_dist_list[t][node_order[t]])
            wandb.log({"log_prob_sigma_t": log_prob.item()})
            ordering_loss = ordering_loss + (reward * log_prob)
            

        return ordering_loss / M

    def train_step(self, batch, M):
        '''
        Performs one training step for both the denoising and diffusion ordering networks.
        '''
        self.denoising_optimizer.zero_grad()
        self.ordering_optimizer.zero_grad()

        # Generate diffusion trajectories for each graph in the batch
        total_denoising_loss = 0
        total_ordering_loss = 0
        for graph in batch:
            graph = self.preprocess(graph)
            diffusion_trajectories = self.generate_diffusion_trajectories(graph, M)

            # Compute denoising loss
            denoising_loss = sum([self.compute_denoising_loss(traj[0], traj[1], traj[2]) for traj in diffusion_trajectories])
            total_denoising_loss += denoising_loss

        # Backpropagation
        total_denoising_loss.backward()
        self.denoising_optimizer.step()
        wandb.log({"denoising_loss": total_denoising_loss.item()})

        for graph in batch:
            graph = self.preprocess(graph)
            diffusion_trajectories = self.generate_diffusion_trajectories(graph, M)

            # Compute ordering loss using REINFORCE
            ordering_loss = self.compute_ordering_loss(diffusion_trajectories, M)
            total_ordering_loss += ordering_loss

        total_ordering_loss.backward()
        self.ordering_optimizer.step()
        wandb.log({"ordering_loss": total_ordering_loss.item()})

        return total_denoising_loss.item(), total_ordering_loss.item()


    def predict_new_node(self, 
                         graph, 
                         sampling_method="sample",
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
            # predict node type
            node_type_probs, edge_type_probs = self.denoising_network(graph.x, graph.edge_index, graph.edge_attr)
            node_type_probs = node_type_probs[-1] # only predict for last node
            
            # sample node type
            if sampling_method == "sample":
                node_type = torch.distributions.Categorical(probs=node_type_probs.squeeze()).sample()
            elif sampling_method == "argmax":
                node_type = torch.argmax(node_type_probs.squeeze(), dim=-1).reshape(-1, 1)

            # sample edge type
            if sampling_method == "sample":
                new_connections = torch.multinomial(edge_type_probs.squeeze(), num_samples=1, replacement=True)
            elif sampling_method == "argmax":
                new_connections = torch.argmax(edge_type_probs.squeeze(), dim=-1).reshape(-1, 1)
            # no need to filter connection to previously denoised nodes, assuming only one new node is added at a time
           
        return node_type, new_connections


    def save_model(self, denoising_network_path="denoising_network.pt", diffusion_ordering_network_path="diffusion_ordering_network.pt"):
        torch.save(self.denoising_network.state_dict(), denoising_network_path)
        torch.save(self.diffusion_ordering_network.state_dict(), diffusion_ordering_network_path)

    def load_model(self,
                   denoising_network_path="denoising_network.pt",
                   diffusion_ordering_network_path="diffusion_ordering_network.pt"):
        self.denoising_network.load_state_dict(torch.load(denoising_network_path, map_location=self.device ))
        self.diffusion_ordering_network.load_state_dict(torch.load(diffusion_ordering_network_path, map_location=self.device))