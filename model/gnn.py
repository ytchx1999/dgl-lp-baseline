import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
import time
import numpy as np
import tqdm
import math


class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers=2, dropout=0.5):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.layers.append(dglnn.GraphConv(in_feats, n_hidden))
        self.bns.append(nn.BatchNorm1d(self.n_hidden))

        for l in range(1, self.n_layers - 1):
            self.layers.append(dglnn.GraphConv(n_hidden, n_hidden))
            self.bns.append(nn.BatchNorm1d(self.n_hidden))

        self.layers.append(dglnn.GraphConv(n_hidden, n_hidden))
        self.predictor = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1))

    def predict(self, h_src, h_dst):
        return self.predictor(h_src * h_dst)

    def forward(self, pair_graph, neg_pair_graph, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.bns[l](h)
                h = F.relu(h)
                h = self.dropout(h)
        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        h_pos = self.predict(h[pos_src], h[pos_dst])
        h_neg = self.predict(h[neg_src], h[neg_dst])
        return h_pos, h_neg

    def inference(self, g, total_nodes, feat, device, batch_size, buffer_device=None):
        # The difference between this inference function and the one in the official
        # example is that the intermediate results can also benefit from prefetching.
        # feat = g.ndata['feat']
        print('Inference...', flush=True)
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1) # prefetch_node_feats=['feat']
        dataloader = dgl.dataloading.DataLoader(
                g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
                batch_size=1000, shuffle=False, drop_last=False, num_workers=0)
        if buffer_device is None:
            buffer_device = device

        for l, layer in enumerate(self.layers):
            # g.num_nodes()  total_nodes
            y = torch.zeros(total_nodes, self.n_hidden, device=buffer_device) # pin_memory=args.pure_gpu
            feat = feat.detach().clone().to(device)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = self.bns[l](h)
                    h = F.relu(h)
                y[output_nodes] = h.to(buffer_device)
            feat = y
        return y


class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers=3, dropout=0.5):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        self.bns.append(nn.BatchNorm1d(self.n_hidden))

        for l in range(1, self.n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
            self.bns.append(nn.BatchNorm1d(self.n_hidden))

        self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.predictor = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1))

    def predict(self, h_src, h_dst):
        return self.predictor(h_src * h_dst)

    def forward(self, pair_graph, neg_pair_graph, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.bns[l](h)
                h = F.relu(h)
                h = self.dropout(h)
        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        h_pos = self.predict(h[pos_src], h[pos_dst])
        h_neg = self.predict(h[neg_src], h[neg_dst])
        return h_pos, h_neg

    def inference(self, g, total_nodes, feat, device, batch_size, buffer_device=None):
        # The difference between this inference function and the one in the official
        # example is that the intermediate results can also benefit from prefetching.
        # feat = g.ndata['feat']
        print('Inference...', flush=True)
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1) # prefetch_node_feats=['feat']
        dataloader = dgl.dataloading.DataLoader(
                g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
                batch_size=1000, shuffle=False, drop_last=False, num_workers=0)
        if buffer_device is None:
            buffer_device = device

        for l, layer in enumerate(self.layers):
            # g.num_nodes()  total_nodes
            y = torch.zeros(total_nodes, self.n_hidden, device=buffer_device) # pin_memory=args.pure_gpu
            feat = feat.detach().clone().to(device)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = self.bns[l](h)
                    h = F.relu(h)
                y[output_nodes] = h.to(buffer_device)
            feat = y
        return y


class DGIConv(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers=3, dropout=0.5):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        self.bns.append(nn.BatchNorm1d(self.n_hidden))

        for l in range(1, self.n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
            self.bns.append(nn.BatchNorm1d(self.n_hidden))

        self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.predictor = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1))

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.bns[l](h)
                h = F.relu(h)
                h = self.dropout(h)
        return h


class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.reset_parameters()

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, features, summary):
        features = torch.matmul(features, torch.matmul(self.weight, summary))
        return features


class DGI(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=3, dropout=0.5):
        super(DGI, self).__init__()
        self.encoder = DGIConv(input_dim, hidden_dim, n_layers, dropout)
        self.discriminator = Discriminator(hidden_dim)
        self.loss = nn.BCEWithLogitsLoss()

    @staticmethod
    def corruption(blocks, x):
        return blocks, x[torch.randperm(x.size(0))]
    
    def predict(self, h_src, h_dst):
        return self.encoder.predictor(h_src * h_dst)

    def forward(self, pair_graph, neg_pair_graph, blocks, x):
        positive = self.encoder(blocks, x)
        negative = self.encoder(*self.corruption(blocks, x))

        summary = torch.sigmoid(positive.mean(dim=0))
        pos = self.discriminator(positive, summary)
        neg = self.discriminator(negative, summary)

        l1 = self.loss(pos, torch.ones_like(pos))
        l2 = self.loss(neg, torch.zeros_like(neg))

        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        h_pos = self.predict(positive[pos_src], positive[pos_dst])
        h_neg = self.predict(positive[neg_src], positive[neg_dst])
        
        return h_pos, h_neg, (l1 + l2)
    
    def inference(self, g, total_nodes, feat, device, batch_size, buffer_device=None):
        # The difference between this inference function and the one in the official
        # example is that the intermediate results can also benefit from prefetching.
        # feat = g.ndata['feat']
        print('Inference...', flush=True)
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1) # prefetch_node_feats=['feat']
        dataloader = dgl.dataloading.DataLoader(
                g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
                batch_size=1000, shuffle=False, drop_last=False, num_workers=0)
        if buffer_device is None:
            buffer_device = device

        for l, layer in enumerate(self.encoder.layers):
            # g.num_nodes()  total_nodes
            y = torch.zeros(total_nodes, self.encoder.n_hidden, device=buffer_device) # pin_memory=args.pure_gpu
            feat = feat.detach().clone().to(device)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)
                if l != len(self.encoder.layers) - 1:
                    h = self.encoder.bns[l](h)
                    h = F.relu(h)
                y[output_nodes] = h.to(buffer_device)
            feat = y
        return y


class GAT(nn.Module):
    def __init__(self, in_feats, n_hidden, n_heads=2, n_layers=3, dropout=0.5):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.layers.append(dglnn.GATConv(in_feats, n_hidden, n_heads))
        self.bns.append(nn.BatchNorm1d(self.n_hidden * n_heads))

        for l in range(1, self.n_layers - 1):
            self.layers.append(dglnn.GATConv(n_hidden * n_heads, n_hidden, n_heads))
            self.bns.append(nn.BatchNorm1d(self.n_hidden * n_heads))

        self.layers.append(dglnn.GATConv(n_hidden * n_heads, n_hidden, 1))
        self.predictor = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1))

    def predict(self, h_src, h_dst):
        return self.predictor(h_src * h_dst)

    def forward(self, pair_graph, neg_pair_graph, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h).flatten(1, -1)
            if l != len(self.layers) - 1:
                h = self.bns[l](h)
                h = F.relu(h)
                h = self.dropout(h)
        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        h_pos = self.predict(h[pos_src], h[pos_dst])
        h_neg = self.predict(h[neg_src], h[neg_dst])
        return h_pos, h_neg

    def inference(self, g, total_nodes, feat, device, batch_size, buffer_device=None):
        # The difference between this inference function and the one in the official
        # example is that the intermediate results can also benefit from prefetching.
        # feat = g.ndata['feat']
        print('Inference...', flush=True)
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1) # prefetch_node_feats=['feat']
        dataloader = dgl.dataloading.DataLoader(
                g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
                batch_size=1000, shuffle=False, drop_last=False, num_workers=0)
        if buffer_device is None:
            buffer_device = device

        for l, layer in enumerate(self.layers):
            # g.num_nodes()  total_nodes
            if l == len(self.layers) - 1:
                y = torch.zeros(total_nodes, self.n_hidden, device=buffer_device) # pin_memory=args.pure_gpu
            else:
                y = torch.zeros(total_nodes, self.n_hidden * self.n_heads, device=buffer_device)
            feat = feat.detach().clone().to(device)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x).flatten(1, -1)
                if l != len(self.layers) - 1:
                    h = self.bns[l](h)
                    h = F.relu(h)
                y[output_nodes] = h.to(buffer_device)
            feat = y
        return y


class GIN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers=3, dropout=0.5, aggr_type='mean', learn_eps=False):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.mlps = nn.ModuleList()

        self.mlps.append(MLP(2, in_feats, n_hidden, n_hidden))
        self.layers.append(dglnn.GINConv(ApplyNodeFunc(self.mlps[0]), aggr_type, 0, learn_eps))
        self.bns.append(nn.BatchNorm1d(self.n_hidden))

        for l in range(1, self.n_layers - 1):
            self.mlps.append(MLP(2, n_hidden, n_hidden, n_hidden))
            self.layers.append(dglnn.GINConv(ApplyNodeFunc(self.mlps[l]), aggr_type, 0, learn_eps))
            self.bns.append(nn.BatchNorm1d(self.n_hidden))

        self.mlps.append(MLP(2, n_hidden, n_hidden, n_hidden))
        self.layers.append(dglnn.GINConv(ApplyNodeFunc(self.mlps[-1]), aggr_type, 0, learn_eps))
        self.predictor = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1))

    def predict(self, h_src, h_dst):
        return self.predictor(h_src * h_dst)

    def forward(self, pair_graph, neg_pair_graph, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.bns[l](h)
                h = F.relu(h)
                h = self.dropout(h)
        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        h_pos = self.predict(h[pos_src], h[pos_dst])
        h_neg = self.predict(h[neg_src], h[neg_dst])
        return h_pos, h_neg

    def inference(self, g, total_nodes, feat, device, batch_size, buffer_device=None):
        # The difference between this inference function and the one in the official
        # example is that the intermediate results can also benefit from prefetching.
        # feat = g.ndata['feat']
        print('Inference...', flush=True)
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1) # prefetch_node_feats=['feat']
        dataloader = dgl.dataloading.DataLoader(
                g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
                batch_size=1000, shuffle=False, drop_last=False, num_workers=0)
        if buffer_device is None:
            buffer_device = device

        for l, layer in enumerate(self.layers):
            # g.num_nodes()  total_nodes
            y = torch.zeros(total_nodes, self.n_hidden, device=buffer_device) # pin_memory=args.pure_gpu
            feat = feat.detach().clone().to(device)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = self.bns[l](h)
                    h = F.relu(h)
                y[output_nodes] = h.to(buffer_device)
            feat = y
        return y


class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """MLP layers construction
        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        """
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class SGC(nn.Module):
    def __init__(self, in_feats, n_hidden, k_hop=3):
        super().__init__()
        self.n_hidden = n_hidden
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SGConv(in_feats, n_hidden, k=k_hop))
        self.predictor = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1))

    def predict(self, h_src, h_dst):
        return self.predictor(h_src * h_dst)

    def forward(self, pair_graph, neg_pair_graph, blocks, x):
        h = x
        layer = self.layers[0]
        block = dgl.add_self_loop(pair_graph)
        h = layer(block, h)
        # for l, (layer, block) in enumerate(zip(self.layers, blocks)):
        #     block = dgl.add_self_loop(block)
        #     h = layer(block, h)
        #     if l != len(self.layers) - 1:
        #         h = F.relu(h)
        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        h_pos = self.predict(h[pos_src], h[pos_dst])
        h_neg = self.predict(h[neg_src], h[neg_dst])
        return h_pos, h_neg

    def inference(self, g, total_nodes, feat, device, batch_size, buffer_device=None):
        # The difference between this inference function and the one in the official
        # example is that the intermediate results can also benefit from prefetching.
        # feat = g.ndata['feat']
        print('Inference...', flush=True)
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1) # prefetch_node_feats=['feat']
        dataloader = dgl.dataloading.DataLoader(
                g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
                batch_size=1000, shuffle=False, drop_last=False, num_workers=0)
        if buffer_device is None:
            buffer_device = device

        for l, layer in enumerate(self.layers):
            # g.num_nodes()  total_nodes
            y = torch.zeros(total_nodes, self.n_hidden, device=buffer_device) # pin_memory=args.pure_gpu
            feat = feat.detach().clone().to(device)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                y[output_nodes] = h.to(buffer_device)
            feat = y
        return y


class VGAEModel(nn.Module):
    def __init__(self, in_dim, hidden1_dim, hidden2_dim, device='cpu'):
        super(VGAEModel, self).__init__()
        self.in_dim = in_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.device = device

        layers = [dglnn.GraphConv(self.in_dim, self.hidden1_dim, activation=F.relu, allow_zero_in_degree=True),
                  dglnn.GraphConv(self.hidden1_dim, self.hidden2_dim, activation=lambda x: x, allow_zero_in_degree=True),
                  dglnn.GraphConv(self.hidden1_dim, self.hidden2_dim, activation=lambda x: x, allow_zero_in_degree=True)]
        self.layers = nn.ModuleList(layers)

    def encoder(self, g, features):
        h = self.layers[0](g, features)
        self.mean = self.layers[1](g, h)
        self.log_std = self.layers[2](g, h)
        gaussian_noise = torch.randn(features.size(0), self.hidden2_dim).to(self.device)
        sampled_z = self.mean + gaussian_noise * torch.exp(self.log_std).to(self.device)
        return sampled_z

    def decoder(self, z):
        adj_rec = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_rec

    def forward(self, g, features):
        z = self.encoder(g, features)
        adj_rec = self.decoder(z)
        return adj_rec
