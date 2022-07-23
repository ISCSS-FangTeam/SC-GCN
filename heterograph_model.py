import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F
from heterographConv_model import HeteroGraphConv

class scgcn(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.conv1 = HeteroGraphConv({
            rel: dglnn.SAGEConv(in_feats, hid_feats, aggregator_type='pool')
            for rel in rel_names}, aggregate='mean')
        self.conv2 = HeteroGraphConv({
            rel: dglnn.SAGEConv(hid_feats, out_feats, aggregator_type='pool')
            for rel in rel_names}, aggregate='mean')

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h

