'''
Test Suite for the NodeMasking class

Tests:
    - masking a single node
    - TODO demasking a single node
    - adding a masked node
    - idxify
    - deidxify
    - is_masked
    - remove_node
    - generate_fully_masked
    - test fully_connect
'''


import pytest
import torch
from torch_geometric.data import Data
from ..utils import NodeMasking

@pytest.fixture
def test_data():
    # Create a mock dataset
    x = torch.tensor([[0], [1], [5]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]], dtype=torch.long)
    edge_attr = torch.tensor([0, 1, 2, 0], dtype=torch.float)
    datapoint = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return datapoint


class TestNodeMasking:
    @pytest.fixture(autouse=True)
    def setup(self, test_data):
        self.test_data = test_data
        self.test_masker = NodeMasking(test_data)

    def test_mask_single_node(self):
        # Test masking one node at a time
        for node in range(self.test_data.x.shape[0]):
            masked_datapoint = self.test_masker.mask_node(self.test_data, node)
            self._assert_node_masked(masked_datapoint, node)
            self._assert_edges_masked(masked_datapoint, node)

    def _assert_node_masked(self, masked_datapoint, node):
        assert masked_datapoint.x[node] == self.test_masker.NODE_MASK

    def _assert_edges_masked(self, masked_datapoint, node):
        assert torch.all(masked_datapoint.edge_attr[masked_datapoint.edge_index[0] == node] == self.test_masker.EDGE_MASK)
        assert torch.all(masked_datapoint.edge_attr[masked_datapoint.edge_index[1] == node] == self.test_masker.EDGE_MASK)

    def test_add_masked_node(self):
        # Test adding a masked node
        masked_datapoint = self.test_masker.add_masked_node(self.test_data)
        self._assert_new_node_added(masked_datapoint)
        self._assert_new_node_masked(masked_datapoint)
        self._assert_new_node_edges_masked(masked_datapoint)
        self._assert_new_node_connected(masked_datapoint)

    def _assert_new_node_added(self, masked_datapoint):
        assert masked_datapoint.x.shape[0] == self.test_data.x.shape[0] + 1

    def _assert_new_node_masked(self, masked_datapoint):
        assert torch.all(masked_datapoint.x[-1] == self.test_masker.NODE_MASK)

    def _assert_new_node_edges_masked(self, masked_datapoint):
        assert torch.all(masked_datapoint.edge_attr[-1] == self.test_masker.EDGE_MASK)

    def _assert_new_node_connected(self, masked_datapoint):
        assert torch.all(masked_datapoint.edge_attr[masked_datapoint.edge_index[0] == masked_datapoint.x.shape[0] - 1] == self.test_masker.EDGE_MASK)

    def test_is_masked(self):
        # Test if a node is masked
        for node in range(self.test_data.x.shape[0]):
            masked_datapoint = self.test_masker.mask_node(self.test_data, node)
            assert self.test_masker.is_masked(masked_datapoint, node)
            demasked_datapoint = self.test_masker.demask_node(masked_datapoint, node, torch.zeros(1), torch.zeros(self.test_data.x.shape[0]))
            assert not self.test_masker.is_masked(demasked_datapoint, node)

    def test_remove_node(self):
        # Test removing a node
        for node in range(self.test_data.x.shape[0]):
            removed_datapoint = self.test_masker.remove_node(self.test_data, node)
            
            # Check if node is removed
            assert removed_datapoint.x.shape[0] == self.test_data.x.shape[0] - 1
            # Check if removed node is not in the new node list
            assert torch.all(removed_datapoint.x == self.test_data.x[torch.arange(self.test_data.x.shape[0]) != node])

            # Check if edges are removed
            assert torch.all(removed_datapoint.edge_index[0] < removed_datapoint.x.shape[0])
            assert torch.all(removed_datapoint.edge_index[1] < removed_datapoint.x.shape[0])
            # Check if dimensions of edge_attr and edge_index are consistent
            assert removed_datapoint.edge_attr.shape[0] == removed_datapoint.edge_index.shape[1]

    def test_generate_fully_masked(self):
        # Test generating a fully masked graph
        masked_datapoint = self.test_masker.generate_fully_masked(n_nodes=5)
        assert torch.all(masked_datapoint.x == self.test_masker.NODE_MASK)
        assert torch.all(masked_datapoint.edge_attr == self.test_masker.EDGE_MASK)

    def test_remove_empty_edges(self):
        # Test removing empty edges
        self.test_data.edge_attr[0:2] = self.test_masker.EMPTY_EDGE
        removed_datapoint = self.test_masker.remove_empty_edges(self.test_data)
        assert removed_datapoint.edge_attr.shape[0] == removed_datapoint.edge_index.shape[1]
        assert torch.all(removed_datapoint.edge_attr != self.test_masker.EMPTY_EDGE)

class TestNodeReIndexing:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.test_data = Data(
            x=torch.tensor([[0], [4], [18]], dtype=torch.float),
            edge_index=torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]], dtype=torch.long),
            edge_attr=torch.tensor([1, 2, 5, 1], dtype=torch.float)
        )
        self.reindexed_data = Data(
            x=torch.tensor([[0], [1], [2]], dtype=torch.float),
            edge_index=torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]], dtype=torch.long),
            edge_attr=torch.tensor([0, 1, 2, 0], dtype=torch.float)
        )
        self.masker = NodeMasking(self.test_data)
    
    def test_idxify(self):
        reindexed_data = self.masker.idxify(self.test_data)
        assert torch.all(reindexed_data.x == self.reindexed_data.x), reindexed_data.x
        assert torch.all(reindexed_data.edge_index == self.reindexed_data.edge_index)
        # assert torch.all(reindexed_data.edge_attr == self.reindexed_data.edge_attr), reindexed_data.edge_attr

    def test_reindex(self):
        reindexed_data = self.masker.idxify(self.test_data)
        reindexed_data = self.masker.deidxify(reindexed_data)
        assert torch.all(reindexed_data.x == self.reindexed_data.x), reindexed_data.x
        assert torch.all(reindexed_data.edge_index == self.reindexed_data.edge_index)
        assert torch.all(reindexed_data.edge_attr == self.reindexed_data.edge_attr), reindexed_data.edge_attr

def test_fully_connect(test_data):
    masker = NodeMasking(test_data)
    fully_connected_graph = masker.fully_connect(test_data)
    assert fully_connected_graph.edge_index.shape[1] == test_data.x.shape[0]**2
    # Assert symmetry in the fully connected graph (edge attributes are the same in both directions)
    for i, j in fully_connected_graph.edge_index.T:
        assert fully_connected_graph.edge_attr[i * fully_connected_graph.x.shape[0] + j] == fully_connected_graph.edge_attr[j * fully_connected_graph.x.shape[0] + i]