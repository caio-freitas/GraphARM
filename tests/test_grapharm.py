'''
Test suite for GraphARM class

- test_predict_single_node: Test if (untrained) GraphArm can predict a single node 
                            correctly for a single-node masked graph

- test_predict_and_add_node: Test if GraphArm can predict a single node and add it to the graph
                            achieving a graph with the correct number of unmasked nodes    

- test_node_decay_ordering: Test if GraphArm can correctly order the nodes in a graph according
                            to the output of an untrained diffusion ordering network

- test_generage_diffusion_trajectories: Test if GraphArm can correctly generate the 
                            graph diffusion trajectories for a given graph   [Requires on test_node_decay_ordering]

- TODO test_loss: Test if GraphArm can correctly calculate the VLB loss for a given graph 
                            and the output of the denoising network and diffusion ordering network

'''
import pytest
import torch
import torch_geometric

from ..grapharm import GraphARM
from ..models import DiffusionOrderingNetwork, DenoisingNetwork
from ..utils import NodeMasking

@pytest.fixture
def test_data():
    # Create a mock dataset
    x = torch.tensor([[0], [1], [5]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]], dtype=torch.long)
    edge_attr = torch.tensor([0, 1, 2, 0], dtype=torch.float)
    datapoint = torch_geometric.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return datapoint

class TestGraphARM:
    @pytest.fixture(autouse=True)
    def setup(self, test_data):
        self.test_data = test_data
        self.test_masker = NodeMasking(test_data)
        denoising_network = DenoisingNetwork(
            node_feature_dim=1,
            edge_feature_dim=1,
            num_node_types=self.test_data.x.unique().shape[0],
            num_edge_types=self.test_data.edge_attr.unique().shape[0],
            device=torch.device('cpu')
            )
        diffusion_ordering_network = DiffusionOrderingNetwork(
            node_feature_dim=1,
            num_edge_types=self.test_data.edge_attr.unique().shape[0],
            num_node_types=self.test_data.x.unique().shape[0],
            device=torch.device('cpu')
            )
        self.grapharm = GraphARM(dataset=test_data, denoising_network=denoising_network, diffusion_ordering_network=diffusion_ordering_network, device=torch.device('cpu'))
        self.empty_graph = self.test_masker.generate_fully_masked(n_nodes=1)

    @pytest.fixture
    def test_predict_single_node(self):
        # Test if GraphARM can predict a single node correctly for a single-node masked graph
        predicted_node_type, predicted_connection_types = self.grapharm.predict_new_node(self.empty_graph, sampling_method='sample', preprocess=False)
        # Assert that predicted node type is a number in the range of node types
        assert predicted_node_type in range(self.test_data.x.unique().shape[0])
        # Assert that predicted connection types are numbers in the range of edge types, +1 for the empty connection type
        assert all([connection_type in range(self.test_data.edge_attr.unique().shape[0]+1) for connection_type in predicted_connection_types])
        return predicted_node_type, predicted_connection_types
    
    def test_predict_and_add_node(self, test_data):
        new_graph = self.test_masker.add_masked_node(test_data)
        # Test if NodeMasking correctly adds the predicted node to the graph
        predicted_node_type, predicted_connection_types = self.grapharm.predict_new_node(new_graph, sampling_method='sample', preprocess=False)
        new_graph = self.test_masker.demask_node(new_graph, new_graph.x.shape[0]-1, predicted_node_type, predicted_connection_types)
        # Assert that the new graph has the new node type
        assert predicted_node_type in new_graph.x
        # Assert that there are no masked edges in the new graph (no occurrence of masker.EDGE_MASK)
        assert self.test_masker.EDGE_MASK not in new_graph.edge_attr

        # Assert that the new graph has the correct number of nodes
        assert new_graph.x.shape[0] == test_data.x.shape[0] + 1

    def test_node_decay_ordering(self):
        test_data = self.test_masker.idxify(self.test_data)
        node_order, sigma_t_dist_list = self.grapharm.node_decay_ordering(test_data)
        # Convert node_order to a list to compare with the set of nodes in the graph
        node_order = [node.item() for node in node_order]
        # Assert that the node order is a permutation of the nodes in the graph
        assert set(node_order) == set(range(self.test_data.x.shape[0]))
        for t in range(len(sigma_t_dist_list)):
            # Assert probabilities sum to 1
            assert torch.allclose(sum(sigma_t_dist_list[t]), torch.tensor(1.0), atol=1e-6)
            # Assert there are t nodes with probability 0 (already masked)
            assert sum([1 for prob in sigma_t_dist_list[t] if prob == 0]) == t

    @pytest.fixture
    def diffusion_trajectory(self):
        test_graph = self.test_masker.idxify(self.test_data)
        test_graph = self.grapharm.preprocess(test_graph)
        diffusion_trajectory, node_order, sigma_t_dist = self.grapharm.generate_diffusion_trajectories(test_graph, M=1)[0]
        return diffusion_trajectory, node_order, sigma_t_dist

    def test_diffusion_trajectory_final_state(self, diffusion_trajectory):
        diffusion_trajectory, _, _ = diffusion_trajectory
        # Check if last graph in diffusion_trajectory is a single-node masked graph
        assert diffusion_trajectory[-1].x.shape[0] == 1
        assert torch.allclose(diffusion_trajectory[-1].edge_index, torch.tensor([[0], [0]]))
        assert torch.allclose(diffusion_trajectory[-1].edge_attr, torch.tensor([self.test_masker.EDGE_MASK]))

    def test_predict_single_node(self):
        # Test if GraphARM can predict a single node correctly for a single-node masked graph
        predicted_node_type, predicted_connection_types = self.grapharm.predict_new_node(self.empty_graph, sampling_method='sample', preprocess=False)
        # Assert that predicted node type is a number in the range of node types (according to the unique node types in the dataset)
        assert predicted_node_type in range(self.test_data.x.unique().shape[0])
        # Assert that predicted connection types are numbers in the range of edge types, +1 for the empty connection type
        assert all([connection_type in range(self.test_data.edge_attr.unique().shape[0]+1) for connection_type in predicted_connection_types])
    
    def test_generate_sample_graph(self):
        # Generate a new graph with 5 nodes
        gen_graph = self.test_masker.generate_fully_masked(n_nodes=1)
        for i in range(4):
            node_type, connections = self.grapharm.predict_new_node(gen_graph, sampling_method='sample', preprocess=False)
            gen_graph = self.test_masker.demask_node(gen_graph, i, node_type, connections)
            gen_graph = self.test_masker.add_masked_node(gen_graph)

        node_type, connections = self.grapharm.predict_new_node(gen_graph, sampling_method='sample', preprocess=False)
        gen_graph = self.test_masker.demask_node(gen_graph, 4, node_type, connections)

        # remove masker.EMPTY_EDGE from edge_attr, and equivalent in edge_index
        gen_graph.edge_index = gen_graph.edge_index[:, gen_graph.edge_attr.squeeze() != self.test_masker.EMPTY_EDGE]
        gen_graph.edge_attr = gen_graph.edge_attr[gen_graph.edge_attr.squeeze() != self.test_masker.EMPTY_EDGE]

        gen_graph = self.test_masker.deidxify(gen_graph)

        # Assert that the generated graph has the correct number of nodes
        assert gen_graph.x.shape[0] == 5
        # Assert that there are no masked nodes in the generated graph
        assert self.test_masker.NODE_MASK not in gen_graph.x
        # Assert that there are no masked edges in the generated graph
        assert self.test_masker.EDGE_MASK not in gen_graph.edge_attr
        

