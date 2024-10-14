
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

        eta = self.ordering_optimizer.param_groups[0]['lr']  # Learning rate from optimizer
        self.denoising_network.train()
        self.diffusion_ordering_network.eval()
        acc_loss = 0.0
        with tqdm(train_data) as pbar:
            for graph in pbar:
                graph = self.preprocess(graph)
                diffusion_trajectories = self.generate_diffusion_trajectories(graph, M)

                # Compute loss based on the simplified training objective
                for diffusion_trajectory, node_order, sigma_t_dist in diffusion_trajectories:
                    G_0 = diffusion_trajectory[0]
                    n = len(node_order)
                    
                    # Uniform sampling of t from 1 to n
                    t = torch.randint(1, n+1, (1,)).item()  # Sample t uniformly from 1 to n

                    # Predict the node and edge types at step t
                    G_pred = diffusion_trajectory[t].clone()
                    node_type_probs, edge_type_probs = self.denoising_network(G_pred.x, G_pred.edge_index, G_pred.edge_attr)

                    # Get the log-likelihood of the prediction
                    for k in range(n - t):
                        w_k = sigma_t_dist[t][node_order[k]].detach()  # Ordering probability for node k
                        loss = self.compute_loss(G_0, node_type_probs, edge_type_probs, w_k, node_order[k], node_order, t, n)
                        acc_loss += loss.item()

                        # Backpropagate the loss
                        loss.backward(retain_graph=True)
                        pbar.set_description(f"Loss: {acc_loss:.4f}")

        # Update parameters
        self.denoising_optimizer.step()
        self.denoising_optimizer.zero_grad()

        # Logging final accumulated loss
        wandb.log({"loss": acc_loss})

        self.denoising_optimizer.zero_grad()
        self.ordering_optimizer.zero_grad()
        
        ## validation batch (optimizing diffusion ordering network)
        self.denoising_network.eval()
        self.diffusion_ordering_network.train()

        reward = torch.tensor(0.0, requires_grad=True)
        acc_reward = 0.0

        # REINFORCE: Accumulate rewards and log probabilities
        reinforce_loss = torch.tensor(0.0, requires_grad=True)

        with tqdm(val_data) as pbar:
            for graph in pbar:
                # preprocess graph
                graph = self.preprocess(graph)
                diffusion_trajectories = self.generate_diffusion_trajectories(graph, M)
                # predictions & loss
                for diffusion_trajectory, node_order, sigma_t_dist_unmasked in diffusion_trajectories:
                    G_0 = diffusion_trajectory[0]
                    for t in range(len(node_order)):
                        for k in range(len(node_order) - t - 1):
                            G_pred = diffusion_trajectory[t + 1].clone()

                            # predict node and edge type distributions
                            node_type_probs, edge_type_probs = self.denoising_network(G_pred.x, G_pred.edge_index, G_pred.edge_attr)
                            # not require grad for output of denoising network
                            node_type_probs = node_type_probs.detach()
                            edge_type_probs = edge_type_probs.detach()

                            w_k = sigma_t_dist_unmasked[t][node_order[k]]
                            # Calculate the reward (negative VLB)
                            reward = self.vlb(G_0, node_type_probs, edge_type_probs, w_k, node_order[k], M)
                            acc_reward += reward.item()

                            # REINFORCE gradient accumulation
                            log_prob = torch.log(sigma_t_dist_unmasked[t][node_order[k]])
                            reinforce_loss = reinforce_loss + reward.detach() * log_prob

                            # Backpropagate (accumulated gradients)
                            pbar.set_description(f"Reward: {acc_reward:.4f}")

        # Update diffusion ordering network using REINFORCE
        reinforce_loss = -(eta / M) * reinforce_loss  # Scale by eta/M as per equation (4)
        reinforce_loss.backward()
        self.ordering_optimizer.step()

        # Log rewards
        wandb.log({"reinforce_loss": reinforce_loss.item(), "reward": acc_reward})
        
    
        
    def compute_loss(self, G_0, node_type_probs, edge_type_probs, w_k, node, node_order, t, n):
        '''
        Computes the log-likelihood for the node and edge type predictions
        as per the simplified training objective in the paper.

        G_0: original graph
        node_type_probs: predicted node type probabilities
        edge_type_probs: predicted edge type probabilities
        w_k: ordering probability for node k
        node: node index
        node_order: node order
        t: current time step
        n: total number of nodes
        '''
        # Calculate the log-likelihood of the node type prediction
        node_type_log_prob = torch.log(node_type_probs[G_0.x[node]])

        # Retrieve edge attributes and calculate the log-likelihood of edge types
        T = len(node_order)
        edge_attrs_matrix = G_0.edge_attr.reshape(T, T)
        original_edge_types = torch.index_select(edge_attrs_matrix[node], 0, torch.tensor(node_order[t:]).to(self.device))
        p_edges = torch.gather(edge_type_probs, 1, original_edge_types.reshape(-1, 1))
        log_p_edges = torch.sum(torch.log(p_edges))

        # Final log-likelihood combining node and edge predictions
        log_p_O_v = node_type_log_prob + log_p_edges

        # Calculate the loss based on the simplified training objective
        loss = -(n / T) * log_p_O_v * w_k / n  # Simplified equation for loss

        return loss

    def vlb(
            self,
            G_0,  # Initial graph (ground truth)
            node_type_probs,  # Predicted node type probabilities (from denoising network)
            edge_type_probs,  # Predicted edge type probabilities (from denoising network)
            w_k,  # Weight of current node in the ordering
            node_index,  # Current node's index in the ordering
            M  # Number of trajectories
        ):


        # True node type from the initial graph (G_0)
        true_node_type = G_0.x[node_index].item()
         # Log-likelihood of node types (cross-entropy)
        node_log_likelihood = torch.log(node_type_probs[true_node_type] + 1e-10)
        node_loss = -w_k * node_log_likelihood

        # Log-likelihood of edge types (cross-entropy)
        # Assuming the edges are stored as tuples in the form (source, target, edge_type)
        edge_loss = 0.0
        for edge in G_0.edge_index.T:
            if edge[0] != node_index:
                continue
            source, target = edge
            true_edge_type = G_0.edge_attr[source * G_0.x.shape[0] + target].item()
            edge_log_likelihood = torch.log(edge_type_probs[target][true_edge_type] + 1e-10)
            edge_loss += -w_k * edge_log_likelihood

        # Combine losses (for the current node and edges at time t)
        loss = (node_loss + edge_loss) / M

        return loss

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