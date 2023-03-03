import math
import logging
from platform import node
import time
import sys
import argparse
from pip import main
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from pathlib import Path
import random

# import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
import time
import numpy as np
from tqdm import tqdm, trange
import os

# from evaluation.evaluation import eval_edge_prediction
from model.gnn import SAGE, SGC
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder, mask_test_edges_dgl
from utils.data_processing import compute_time_statistics, get_data_no_label

from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)
    dgl.seed(seed)


def main():
    ### Argument and global variables
    parser = argparse.ArgumentParser('Link Prediction')
    parser.add_argument('-d', '--data', type=str, help='Dataset name', default='gowalla_Entertainment')
    parser.add_argument('--bs', type=int, default=512, help='Batch_size')
    parser.add_argument('--n_heads', type=int, default=3, help='Number of heads used in attention layer')
    parser.add_argument('--n_epoch', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate') #0.0001
    parser.add_argument('--drop', type=float, default=0.5, help='Dropout')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
    parser.add_argument('--model', type=str, default="graphsage", choices=["graphsage", "sgc", "gcn", "gin", "gat"], help='Type of embedding module')
    parser.add_argument('--n_hidden', type=int, default=256, help='Dimensions of the hidden')
    parser.add_argument("--fanout", type=str, default='15,10,5', help='Neighbor sampling fanout')
    # parser.add_argument("--fanout_sgc", type=str, default='0', help='SGC neighbor sampling fanout')
    parser.add_argument('--different_new_nodes', action='store_true', help='Whether to use disjoint set of new nodes for train and val')
    parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')
    parser.add_argument('--randomize_features', action='store_true', help='Whether to randomize node features')
    parser.add_argument('--data_type', type=str, default="gowalla", help='Type of dataset')
    parser.add_argument('--task_type', type=str, default="time_trans", help='Type of task')
    parser.add_argument('--mode', type=str, default="pretrain", help='pretrain or downstream')
    parser.add_argument('--seed', type=int, default=0, help='Seed for all')
    parser.add_argument('--k_hop', type=int, default=3, help='K-hop for SGC')
    parser.add_argument('--learn_eps', action="store_true", help='learn the epsilon weighting')
    parser.add_argument('--aggr_type', type=str, default="mean", choices=["sum", "mean", "max"], help='type of neighboring pooling: sum, mean or max')

    args = parser.parse_args()
    set_seed(args.seed)

    g, full_g, n_feats, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = get_data_no_label(args.data,
                              different_new_nodes_between_val_and_test=args.different_new_nodes, randomize_features=args.randomize_features, \
                              have_edge=False, data_type=args.data_type, task_type=args.task_type, mode=args.mode, seed=args.seed)
    
    train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

    # Initialize validation and test neighbor finder to retrieve temporal graph
    full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

    # Initialize negative samplers. Set seeds for validation and testing so negatives are the same
    # across different runs
    # NB: in the inductive setting, negatives are sampled only amongst other new nodes
    train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
    val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=args.seed)
    nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations, seed=args.seed)
    test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=args.seed)
    nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources, new_node_test_data.destinations, seed=args.seed)
    # Set device
    device_string = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_string)

    # undirected
    g = dgl.to_bidirected(g)
    full_g = dgl.to_bidirected(full_g)
    num_edges_no_self = g.num_edges()

    if args.model in ['gcn', 'gat']:
        g = dgl.add_self_loop(g)

    g = g.to(device)
    full_g = full_g.to(device)
    train_seed_edges = torch.arange(num_edges_no_self).to(device)

    test_ap_list = []
    test_auc_list = []
    test_f1_micro_list = []
    test_f1_macro_list = []

    for i in range(args.n_runs):
        set_seed(i)
        print('-'*50, flush=True)
        print(f'Run {i}:', flush=True)

        if args.mode == 'pretrain':
            node_features = torch.nn.Parameter(torch.from_numpy(n_feats).to(torch.float32)).to(device)
        else:
            node_features = torch.nn.Parameter(torch.load('./results/emb_{}_{}_pretrain.pth'.format(args.data, args.task_type), map_location='cpu')).to(device)

        if args.model == 'sgc':
            model = SGC(node_features.shape[1], args.n_hidden, args.k_hop, args.drop)

        if not (args.mode == 'pretrain'):
            ckpt = torch.load('./results/model_{}_{}_{}_pretrain.pth'.format(args.model, args.data, args.task_type), map_location='cpu')
            model.load_state_dict(ckpt, strict=False)
        
        model = model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        best_val_ap = 0.
        best_test_ap = 0.
        best_test_result = None
        best_nn_test_result = None

        for epoch in range(args.n_epoch):
            ap, f1, auc, m_loss = [], [], [], []

            model.train()
            logits = model.forward(g, node_features)   # here

            # compute loss
            loss = norm * F.binary_cross_entropy(logits.view(-1), adj.view(-1), weight=weight_tensor)
            kl_divergence = 0.5 / logits.size(0) * (1 + 2 * model.log_std - model.mean ** 2 - torch.exp(model.log_std) ** 2).sum(1).mean()
            loss -= kl_divergence

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_acc = get_acc(logits, adj)

            print('Run {}, epoch: {}'.format(i, epoch))
            print('Epoch mean loss: {}'.format(loss.item()))
            print('train acc: {}'.format(train_acc))

            # eval
            if epoch % 5 == 0 or (epoch + 1) == args.n_epoch:
                model.eval()
                val_res = get_scores(val_edges, val_edges_false, logits)
                test_res = get_scores(test_edges, test_edges_false, logits)

                print('*'*50, flush=True)
                print('valid ap: {}'.format(val_res['ap']), flush=True)
                print('valid auc: {}'.format(val_res['auc']), flush=True)
                print('valid f1_micro: {}'.format(val_res['f1_micro']), flush=True)
                print('valid f1_macro: {}'.format(val_res['f1_macro']), flush=True)
                print('*'*50, flush=True)
                print('test ap: {}'.format(test_res['ap']), flush=True)
                print('test auc: {}'.format(test_res['auc']), flush=True)
                print('test f1_micro: {}'.format(test_res['f1_micro']), flush=True)
                print('test f1_macro: {}'.format(test_res['f1_macro']), flush=True)
                print('*'*50, flush=True)

                if best_val_ap < val_res['ap']:
                    best_val_ap = val_res['ap']
                    best_test_result = test_res.copy()

                    if not os.path.exists('./results'):
                        os.makedirs('./results', exist_ok=True)
                    if args.mode == 'pretrain':
                        torch.save(model.state_dict(), './results/model_{}_{}_{}_{}.pth'.format(args.model, args.data, args.task_type, args.mode))
            
        test_ap_list.append(best_test_result['ap'])
        test_auc_list.append(best_test_result['auc'])
        test_f1_micro_list.append(best_test_result['f1_micro'])
        test_f1_macro_list.append(best_test_result['f1_macro'])
    
    # print final results: mean ± std
    best_result_ap_mean, best_result_ap_std = np.mean(np.array(test_ap_list), axis=0), np.std(np.array(test_ap_list), axis=0)
    best_result_auc_mean, best_result_auc_std = np.mean(np.array(test_auc_list), axis=0), np.std(np.array(test_auc_list), axis=0)
    best_result_f1_micro_mean, best_result_f1_micro_std = np.mean(np.array(test_f1_micro_list), axis=0), np.std(np.array(test_f1_micro_list), axis=0)
    best_result_f1_macro_mean, best_result_f1_macro_std = np.mean(np.array(test_f1_macro_list), axis=0), np.std(np.array(test_f1_macro_list), axis=0)
    print(f'Final test ap: {best_result_ap_mean} ± {best_result_ap_std}', flush=True)
    print(f'Final test auc: {best_result_auc_mean} ± {best_result_auc_std}', flush=True)
    print(f'Final test f1_micro: {best_result_f1_micro_mean} ± {best_result_f1_micro_std}', flush=True)
    print(f'Final test f1_macro: {best_result_f1_macro_mean} ± {best_result_f1_macro_std}', flush=True)
    print('-'*50, flush=True)
    # done !


def compute_loss_para(adj, device):
    pos_weight = ((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    weight_mask = adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)).to(device)
    weight_tensor[weight_mask] = pos_weight
    return weight_tensor, norm


def get_acc(adj_rec, adj_label):
    labels_all = adj_label.view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy


def get_scores(edges_pos, edges_neg, adj_rec):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    adj_rec = adj_rec.cpu()
    # Predict on test set of edges
    preds = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]].item()))

    preds_neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    f1_micro = f1_score(labels_all, np.where(preds_all > 0.5, 1, 0), average='micro')
    f1_macro = f1_score(labels_all, np.where(preds_all > 0.5, 1, 0), average='macro')

    return {'ap': ap_score, 'auc': roc_score, 'f1_micro': f1_micro, 'f1_macro': f1_macro}


if __name__ == '__main__':
    main()