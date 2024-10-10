'''
Test suite for GraphARM class

- test_predict_single_node: Test if (untrained) GraphArm can predict a single node 
                            correctly for a single-node masked graph

'''

import torch
import torch_geometric

from ..grapharm import GraphARM
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
        self.grapharm = GraphARM(dataset=test_data, denoising_network=None, diffusion_ordering_network=None, device=torch.device('cpu'))
        self.empty_graph = self.test_masker.generate_fully_masked(n_nodes=1)

    def test_predict_single_node(self):
        # Test if GraphARM can predict a single node correctly for a single-node masked graph
        predicted_node_type, predicted_connection_types = self.grapharm.predict(self.empty_graph)
        assert predicted_node_type.shape[1] == self.empty_graph.x.shape[1]