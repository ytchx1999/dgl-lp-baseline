import math
import logging
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

import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
import time
import numpy as np
import tqdm

# from evaluation.evaluation import eval_edge_prediction
from model.gnn import SAGE
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics, get_data_no_label

from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


def main():
    ### Argument and global variables
    parser = argparse.ArgumentParser('Link Prediction')
    parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)', default='gowalla')
    parser.add_argument('--bs', type=int, default=512, help='Batch_size')
    parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
    parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
    parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
    parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
    parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate') #0.0001
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--patience', type=int, default=3, help='Patience for early stopping')
    parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
    parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
    parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
    parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to backprop')
    parser.add_argument('--model', type=str, default="graphsage", choices=["graphsage", "gat", "gin"], help='Type of embedding module')
    parser.add_argument('--n_hidden', type=int, default=256, help='Dimensions of the hidden')
    parser.add_argument("--fanout", type=str, default='15,10,5')
    parser.add_argument('--different_new_nodes', action='store_true', help='Whether to use disjoint set of new nodes for train and val')
    parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')
    parser.add_argument('--randomize_features', action='store_true', help='Whether to randomize node features')
    parser.add_argument('--use_destination_embedding_in_message', action='store_true', help='Whether to use the embedding of the destination node as part of the message')
    parser.add_argument('--use_source_embedding_in_message', action='store_true', help='Whether to use the embedding of the source node as part of the message')
    parser.add_argument('--k_hop', type=int, default=2, help='hops in the sampled subgraph')
    parser.add_argument('--data_type', type=str, default="gowalla", help='Type of dataset')
    parser.add_argument('--task_type', type=str, default="time_trans", help='Type of task')

    args = parser.parse_args()

    g, node_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = get_data_no_label(args.data,
                              different_new_nodes_between_val_and_test=args.different_new_nodes, randomize_features=args.randomize_features, \
                              have_edge=False, data_type=args.data_type, task_type=args.task_type)
    
    train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

    # Initialize validation and test neighbor finder to retrieve temporal graph
    full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

    # Initialize negative samplers. Set seeds for validation and testing so negatives are the same
    # across different runs
    # NB: in the inductive setting, negatives are sampled only amongst other new nodes
    train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
    val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
    nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations, seed=1)
    test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
    nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources, new_node_test_data.destinations, seed=3)
    # Set device
    device_string = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_string)

    g = dgl.to_bidirected(g, copy_ndata=True)
    g = dgl.add_self_loop(g)

    g = g.to(device)
    train_seed_edges = torch.from_numpy(train_data.edge_idxs).to(device)
    node_features = torch.nn.Parameter(torch.from_numpy(node_features)).to(device)

    test_ap_list = []
    test_auc_list = []
    test_f1_micro_list = []
    test_f1_macro_list = []
    nn_test_ap_list = []
    nn_test_auc_list = []
    nn_test_f1_micro_list = []
    nn_test_f1_macro_list = []

    for i in range(args.n_runs):
        if args.model == 'graphsage':
            model = SAGE(node_features.shape[1], args.n_hidden).to(device)
        
        opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        # criterion = torch.nn.BCELoss()

        fanout = [int(i) for i in args.fanout.split(',')]
        sampler = dgl.dataloading.NeighborSampler(fanout)  # , prefetch_node_feats=['feat']  [15, 10, 5]
        sampler = dgl.dataloading.as_edge_prediction_sampler(
                sampler, 
                negative_sampler=dgl.dataloading.negative_sampler.Uniform(1))  # , exclude='reverse_id', reverse_eids=reverse_eids
        dataloader = dgl.dataloading.DataLoader(
                g, train_seed_edges, sampler,
                device=device, batch_size=args.bs, shuffle=False,
                drop_last=False, num_workers=0)  #  use_uva=not args.pure_gpu
        
        best_val_ap = 0.
        best_val_result = None
        best_nn_val_result = None
        best_test_result = None
        best_nn_test_result = None

        for epoch in range(args.n_epoch):
            ap, f1, auc, m_loss = [], [], [], []

            for it, (input_nodes, pair_graph, neg_pair_graph, blocks) in enumerate(dataloader):
                model.train()

                # x = blocks[0].srcdata['feat']
                x = node_features[input_nodes].to(device)
                pos_score, neg_score = model(pair_graph, neg_pair_graph, blocks, x)
                pos_label = torch.ones_like(pos_score)
                neg_label = torch.zeros_like(neg_score)
                score = torch.cat([pos_score, neg_score])
                labels = torch.cat([pos_label, neg_label])
                loss = F.binary_cross_entropy_with_logits(score, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()

                with torch.no_grad():
                    model = model.eval()
                    pred_score = np.concatenate([(pos_score.sigmoid()).cpu().detach().numpy(), (neg_score.sigmoid()).cpu().detach().numpy()])
                    pred_label = pred_score > 0.5
                    ap.append(average_precision_score(labels.astype(np.float64), pred_score.astype(np.float64)))
                    f1.append(f1_score(labels.astype(np.float64), pred_label.astype(np.float64)))
                    auc.append(roc_auc_score(labels.astype(np.float64), pred_score.astype(np.float64)))
                    m_loss.append(loss.item())
            
            print('Run {}, epoch: {}'.format(i, epoch))
            print('Epoch mean loss: {}'.format(np.mean(m_loss)))
            print('train auc: {}'.format(np.mean(auc)))
            print('train f1: {}'.format(np.mean(f1)))
            print('train ap: {}'.format(np.mean(ap)))

            # eval
            if epoch % 10 == 0 or (epoch + 1) == args.n_epoch:
                model.eval()
                infer_data = val_data, test_data, new_node_val_data, new_node_test_data, val_rand_sampler, nn_val_rand_sampler, test_rand_sampler, nn_test_rand_sampler
                val_res, nn_val_res, test_res, nn_test_res = evaluate(g, model, node_features, device, infer_data)
                
                print('valid ap: {}, new node val ap: {}'.format(val_res['ap'], nn_val_res['ap']), flush=True)
                print('valid auc: {}, new node val auc: {}'.format(val_res['auc'], nn_val_res['auc']), flush=True)
                print('valid f1_micro: {}, new node val f1_micro: {}'.format(val_res['f1_micro'], nn_val_res['f1_micro']), flush=True)
                print('valid f1_macro: {}, new node val f1_macro: {}'.format(val_res['f1_macro'], nn_val_res['f1_macro']), flush=True)

                print('test ap: {}, new node test ap: {}'.format(test_res['ap'], nn_test_res['ap']), flush=True)
                print('test auc: {}, new node test auc: {}'.format(test_res['auc'], nn_test_res['auc']), flush=True)
                print('test f1_micro: {}, new node test f1_micro: {}'.format(test_res['f1_micro'], nn_test_res['f1_micro']), flush=True)
                print('test f1_macro: {}, new node test f1_macro: {}'.format(test_res['f1_macro'], nn_test_res['f1_macro']), flush=True)

                if best_val_ap < val_res['ap']:
                    best_val_ap = val_res['ap']
                    best_test_result = test_res.copy()
                    best_nn_test_result = nn_test_res.copy()
                
        test_ap_list.append(best_test_result['ap'])
        test_auc_list.append(best_test_result['auc'])
        test_f1_micro_list.append(best_test_result['f1_micro'])
        test_f1_macro_list.append(best_test_result['f1_micro'])

        nn_test_ap_list.append(best_nn_test_result['ap'])
        nn_test_auc_list.append(best_nn_test_result['auc'])
        nn_test_f1_micro_list.append(best_nn_test_result['f1_micro'])
        nn_test_f1_macro_list.append(best_nn_test_result['f1_micro'])
    
    # print final results: mean ± std
    best_result_ap_mean, best_result_ap_std = np.mean(np.array(test_ap_list), axis=0), np.std(np.array(test_ap_list), axis=0)
    best_result_auc_mean, best_result_auc_std = np.mean(np.array(test_auc_list), axis=0), np.std(np.array(test_auc_list), axis=0)
    best_result_f1_micro_mean, best_result_f1_micro_std = np.mean(np.array(test_f1_micro_list), axis=0), np.std(np.array(test_f1_micro_list), axis=0)
    best_result_f1_macro_mean, best_result_f1_macro_std = np.mean(np.array(test_f1_macro_list), axis=0), np.std(np.array(test_f1_macro_list), axis=0)
    print(f'Final test ap: {best_result_ap_mean} ± {best_result_ap_std}', flush=True)
    print(f'Final test auc: {best_result_auc_mean} ± {best_result_auc_std}', flush=True)
    print(f'Final test f1_micro: {best_result_f1_micro_mean} ± {best_result_f1_micro_std}', flush=True)
    print(f'Final test f1_macro: {best_result_f1_macro_mean} ± {best_result_f1_macro_std}', flush=True)

    nn_best_result_ap_mean, nn_best_result_ap_std = np.mean(np.array(nn_test_ap_list), axis=0), np.std(np.array(nn_test_ap_list), axis=0)
    nn_best_result_auc_mean, nn_best_result_auc_std = np.mean(np.array(nn_test_auc_list), axis=0), np.std(np.array(nn_test_auc_list), axis=0)
    nn_best_result_f1_micro_mean, nn_best_result_f1_micro_std = np.mean(np.array(nn_test_f1_micro_list), axis=0), np.std(np.array(nn_test_f1_micro_list), axis=0)
    nn_best_result_f1_macro_mean, nn_best_result_f1_macro_std = np.mean(np.array(nn_test_f1_macro_list), axis=0), np.std(np.array(nn_test_f1_macro_list), axis=0)
    print(f'Final new node test ap: {nn_best_result_ap_mean} ± {nn_best_result_ap_std}', flush=True)
    print(f'Final new node test auc: {nn_best_result_auc_mean} ± {nn_best_result_auc_std}', flush=True)
    print(f'Final new node test f1_micro: {nn_best_result_f1_micro_mean} ± {nn_best_result_f1_micro_std}', flush=True)
    print(f'Final new node test f1_macro: {nn_best_result_f1_macro_mean} ± {nn_best_result_f1_macro_std}', flush=True)
    # done !


def compute_mrr(model, node_emb, src, dst, neg_dst, device, batch_size=500):
    rr = torch.zeros(src.shape[0])
    for start in tqdm.trange(0, src.shape[0], batch_size):
        end = min(start + batch_size, src.shape[0])
        all_dst = torch.cat([dst[start:end, None], neg_dst[start:end]], 1)
        h_src = node_emb[src[start:end]][:, None, :].to(device)
        h_dst = node_emb[all_dst.view(-1)].view(*all_dst.shape, -1).to(device)
        pred = model.predict(h_src, h_dst).squeeze(-1)
        relevance = torch.zeros(*pred.shape, dtype=torch.bool).to(pred.device)
        relevance[:, 0] = True
        rr[start:end] = MF.retrieval_reciprocal_rank(pred, relevance)
    return rr.mean()


def compute_metrics(model, node_emb, src, dst, neg_dst, device, batch_size=500):
    # rr = torch.zeros(src.shape[0])
    ap, auc = [], []
    f1_micro, f1_macro = [], []
    for start in tqdm.trange(0, src.shape[0], batch_size):
        end = min(start + batch_size, src.shape[0])
        all_dst = torch.cat([dst[start:end, None], neg_dst[start:end]], 1)
        h_src = node_emb[src[start:end]][:, None, :].to(device)
        h_dst = node_emb[all_dst.view(-1)].view(*all_dst.shape, -1).to(device)
        pred = model.predict(h_src, h_dst).squeeze(-1)
        relevance = torch.zeros(*pred.shape, dtype=torch.bool).to(pred.device)
        relevance[:, 0] = True
        # rr[start:end] = MF.retrieval_reciprocal_rank(pred, relevance)
        pred = pred.numpy()
        relevance = relevance.numpy()
        ap.append(average_precision_score(relevance, pred))  # ? .sigmoid() ?
        auc.append(roc_auc_score(relevance, pred))
        f1_micro.append(f1_score(relevance, np.where(pred > 0.5, 1, 0), average='micro'))
        f1_macro.append(f1_score(relevance, np.where(pred > 0.5, 1, 0), average='macro'))
    return {'ap': np.mean(ap), 'auc': np.mean(auc), 'f1_micro': np.mean(f1_micro), 'f1_macro': np.mean(f1_macro)}


def evaluate(g, model, feat, device, infer_data, num_workers=0):
    with torch.no_grad():
        node_emb = model.inference(g, feat, device, 4096, num_workers, 'cpu')
        val_data, test_data, new_node_val_data, new_node_test_data, val_rand_sampler, nn_val_rand_sampler, test_rand_sampler, nn_test_rand_sampler = infer_data

        val_src, val_dst = val_data.sources, val_data.destinations
        _, val_neg = val_rand_sampler.sample(len(val_src))
        nn_val_src, nn_val_dst = new_node_val_data.sources, new_node_val_data.destinations
        _, nn_val_neg = nn_val_rand_sampler.sample(len(nn_val_src))
        
        test_src, test_dst = test_data.sources, test_data.destinations
        _, test_neg = test_rand_sampler.sample(len(test_src))
        nn_test_src, nn_test_dst = new_node_test_data.sources, new_node_test_data.destinations
        _, nn_test_neg = nn_test_rand_sampler.sample(len(nn_test_src))

        val_src, val_dst, val_neg = torch.from_numpy(val_src).to(node_emb.device), torch.from_numpy(val_dst).to(node_emb.device), torch.from_numpy(val_neg).to(node_emb.device)
        nn_val_src, nn_val_dst, nn_val_neg = torch.from_numpy(nn_val_src).to(node_emb.device), torch.from_numpy(nn_val_dst).to(node_emb.device), torch.from_numpy(nn_val_neg).to(node_emb.device)

        test_src, test_dst, test_neg = torch.from_numpy(test_src).to(node_emb.device), torch.from_numpy(test_dst).to(node_emb.device), torch.from_numpy(test_neg).to(node_emb.device)
        nn_test_src, nn_test_dst, nn_test_neg = torch.from_numpy(nn_test_src).to(node_emb.device), torch.from_numpy(nn_test_dst).to(node_emb.device), torch.from_numpy(nn_test_neg).to(node_emb.device)

        # val_ap, val_auc, nn_val_ap, nn_val_auc = [], [], [], []
        # val_f1_micro, val_f1_macro, nn_val_f1_micro, nn_val_f1_macro = [], [], [], []

        # test_ap, test_auc, nn_test_ap, nn_test_auc = [], [], [], []
        # test_f1_micro, test_f1_macro, nn_test_f1_micro, nn_test_f1_macro = [], [], [], []
        
        val_res = compute_metrics(model, node_emb, val_src, val_dst, val_neg, device)
        nn_val_res = compute_metrics(model, node_emb, nn_val_src, nn_val_dst, nn_val_neg, device)
        test_res = compute_metrics(model, node_emb, test_src, test_dst, test_neg, device)
        nn_test_res = compute_metrics(model, node_emb, test_src, test_dst, test_neg, device)

        # results = []
        # for split in ['valid', 'test']:
        #     src = edge_split[split]['source_node'].to(node_emb.device)
        #     dst = edge_split[split]['target_node'].to(node_emb.device)
        #     neg_dst = edge_split[split]['target_node_neg'].to(node_emb.device)
        #     results.append(compute_mrr(model, node_emb, src, dst, neg_dst, device))
    return val_res, nn_val_res, test_res, nn_test_res


if __name__ == '__main__':
    main()