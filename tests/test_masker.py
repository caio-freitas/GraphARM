import pytest
import torch
from torch_geometric.data import Data
from utils import NodeMasking

@pytest.fixture
def test_data():
    # Create a mock dataset
    x = torch.tensor([[0], [1], [2]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]], dtype=torch.long)
    edge_attr = torch.tensor([0, 1, 2, 0], dtype=torch.float)
    datapoint = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return datapoint

@pytest.fixture
def test_masker(test_data):
    return NodeMasking(test_data)

def test_mask_single_node(test_data, test_masker):
    # Test masking a single node
    for node in range(test_data.x.shape[0]):
        masked_datapoint = test_masker.mask_node(test_data, node)
        assert masked_datapoint.x[node] == test_masker.NODE_MASK
        assert torch.all(masked_datapoint.edge_attr[masked_datapoint.edge_index[0] == node] == test_masker.EDGE_MASK)
        assert torch.all(masked_datapoint.edge_attr[masked_datapoint.edge_index[1] == node] == test_masker.EDGE_MASK)

def test_add_masked_node(test_data, test_masker):
    # Test adding a masked node
    masked_datapoint = test_masker.add_masked_node(test_data)
    assert torch.all(masked_datapoint.x[-1] == test_masker.NODE_MASK)
    assert torch.all(masked_datapoint.edge_attr[-1] == test_masker.EDGE_MASK)
    assert torch.all(masked_datapoint.edge_attr[:-1] == test_data.edge_attr)
    assert torch.all(masked_datapoint.edge_index[:, :-1] == test_data.edge_index)