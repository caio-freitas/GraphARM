'''
Test suite for GraphARM class

- test_predict_single_node: Test if (untrained) GraphArm can predict a single node 
                            correctly for a single-node masked graph

- test_predict_and_add_node: Test if GraphArm can predict a single node and add it to the graph
                            achieving a graph with the correct number of unmasked nodes    

- TODO test_node_decay_ordering: Test if GraphArm can correctly order the nodes in a graph according
                            to the output of an untrained diffusion ordering network

- TODO test_generage_diffusion_trajectories: Test if GraphArm can correctly generate the 
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