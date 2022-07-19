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
from model.gnn import SAGE, GCN, GAT, GIN
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
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
    parser.add_argument('--n_heads', type=int, default=2, help='Number of heads used in attention layer')
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
    # parser.add_argument('--k_hop', type=int, default=3, help='K-hop for SGC')
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
    nn_test_ap_list = []
    nn_test_auc_list = []
    nn_test_f1_micro_list = []
    nn_test_f1_macro_list = []

    for i in range(args.n_runs):
        set_seed(i)
        print('-'*50, flush=True)
        print(f'Run {i}:', flush=True)

        if args.mode == 'pretrain':
            node_features = torch.nn.Parameter(torch.from_numpy(n_feats).to(torch.float32)).to(device)
        else:
            node_features = torch.nn.Parameter(torch.load('./results/emb_{}_{}_pretrain.pth'.format(args.data, args.task_type), map_location='cpu')).to(device)
        
        fanout = [int(i) for i in args.fanout.split(',')]
        n_layers = len(fanout)

        if args.model == 'graphsage':
            model = SAGE(node_features.shape[1], args.n_hidden, n_layers, args.drop)
        elif args.model == 'gcn':
            model = GCN(node_features.shape[1], args.n_hidden, n_layers, args.drop)
        elif args.model == 'gat':
            model = GAT(node_features.shape[1], args.n_hidden, args.n_heads, n_layers, args.drop)
        elif args.model == 'gin':
            model = GIN(node_features.shape[1], args.n_hidden, n_layers, args.drop, args.aggr_type, args.learn_eps)
        # elif args.model == 'sgc':
        #     model = SGC(node_features.shape[1], args.n_hidden, args.k_hop)

        if not (args.mode == 'pretrain'):
            ckpt = torch.load('./results/model_{}_{}_{}_pretrain.pth'.format(args.model, args.data, args.task_type), map_location='cpu')
            model.load_state_dict(ckpt, strict=False)
        
        model = model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # criterion = torch.nn.BCELoss()

        # fanout_sgc = [int(i) for i in args.fanout_sgc.split(',')]
        # if args.model == 'sgc':
        #     sampler = dgl.dataloading.NeighborSampler(fanout_sgc)
        # else:
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

            for input_nodes, pair_graph, neg_pair_graph, blocks in tqdm(dataloader, desc=f"Run {i}, Epoch {epoch}"):
                model.train()

                # x = blocks[0].srcdata['feat']
                x = node_features[input_nodes].to(device)
                pos_score, neg_score = model(pair_graph, neg_pair_graph, blocks, x)
                pos_label = torch.ones_like(pos_score)
                neg_label = torch.zeros_like(neg_score)
                score = torch.cat([pos_score, neg_score]).squeeze(-1)
                labels = torch.cat([pos_label, neg_label]).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(score, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()

                with torch.no_grad():
                    model = model.eval()
                    # pred_score = np.concatenate([(pos_score.sigmoid()).cpu().detach().numpy(), (neg_score.sigmoid()).cpu().detach().numpy()])
                    score = score.sigmoid().cpu().detach().numpy()
                    labels = labels.cpu().detach().numpy()
                    pred_label = score > 0.5
                    ap.append(average_precision_score(labels, score))
                    f1.append(f1_score(labels, pred_label))
                    auc.append(roc_auc_score(labels, score))
                    m_loss.append(loss.item())
                
                # break # only for debug
            
            print('Run {}, epoch: {}'.format(i, epoch))
            print('Epoch mean loss: {}'.format(np.mean(m_loss)))
            print('train auc: {}'.format(np.mean(auc)))
            print('train f1: {}'.format(np.mean(f1)))
            print('train ap: {}'.format(np.mean(ap)))

            # eval
            if epoch % 5 == 0 or (epoch + 1) == args.n_epoch:
                model.eval()
                infer_data = val_data, test_data, new_node_val_data, new_node_test_data, val_rand_sampler, nn_val_rand_sampler, test_rand_sampler, nn_test_rand_sampler
                val_res, nn_val_res, test_res, nn_test_res = evaluate(g, full_g.num_nodes(), model, node_features, device, infer_data)  # full_g  or g? 
                
                print('*'*50, flush=True)
                print('valid ap: {}, new node val ap: {}'.format(val_res['ap'], nn_val_res['ap']), flush=True)
                print('valid auc: {}, new node val auc: {}'.format(val_res['auc'], nn_val_res['auc']), flush=True)
                print('valid f1_micro: {}, new node val f1_micro: {}'.format(val_res['f1_micro'], nn_val_res['f1_micro']), flush=True)
                print('valid f1_macro: {}, new node val f1_macro: {}'.format(val_res['f1_macro'], nn_val_res['f1_macro']), flush=True)
                print('*'*50, flush=True)
                print('test ap: {}, new node test ap: {}'.format(test_res['ap'], nn_test_res['ap']), flush=True)
                print('test auc: {}, new node test auc: {}'.format(test_res['auc'], nn_test_res['auc']), flush=True)
                print('test f1_micro: {}, new node test f1_micro: {}'.format(test_res['f1_micro'], nn_test_res['f1_micro']), flush=True)
                print('test f1_macro: {}, new node test f1_macro: {}'.format(test_res['f1_macro'], nn_test_res['f1_macro']), flush=True)
                print('*'*50, flush=True)

                if best_val_ap < val_res['ap']:
                    best_val_ap = val_res['ap']
                    best_test_result = test_res.copy()
                    best_nn_test_result = nn_test_res.copy()
                    
                    if not os.path.exists('./results'):
                        os.makedirs('./results', exist_ok=True)
                    if args.mode == 'pretrain':
                        torch.save(model.state_dict(), './results/model_{}_{}_{}_{}.pth'.format(args.model, args.data, args.task_type, args.mode))
                        torch.save(node_features.cpu().detach(), './results/emb_{}_{}_{}.pth'.format(args.data, args.task_type, args.mode))
                
        test_ap_list.append(best_test_result['ap'])
        test_auc_list.append(best_test_result['auc'])
        test_f1_micro_list.append(best_test_result['f1_micro'])
        test_f1_macro_list.append(best_test_result['f1_macro'])

        nn_test_ap_list.append(best_nn_test_result['ap'])
        nn_test_auc_list.append(best_nn_test_result['auc'])
        nn_test_f1_micro_list.append(best_nn_test_result['f1_micro'])
        nn_test_f1_macro_list.append(best_nn_test_result['f1_macro'])
    
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
    print(f'Model {args.model}.')
    print('-'*50, flush=True)
    # done !


def compute_metrics(model, node_emb, src, dst, neg_dst, device, batch_size=500):
    # rr = torch.zeros(src.shape[0])
    ap, auc = [], []
    f1_micro, f1_macro = [], []
    for start in trange(0, src.shape[0], batch_size):
        end = min(start + batch_size, src.shape[0])
        h_src = node_emb[src[start:end]].to(device)
        h_dst = node_emb[dst[start:end]].to(device)
        h_neg = node_emb[neg_dst[start:end]].to(device)

        pos_prob = model.predict(h_src, h_dst).squeeze(-1)
        neg_prob = model.predict(h_src, h_neg).squeeze(-1)

        pred_score = np.concatenate([(pos_prob.sigmoid()).cpu().detach().numpy(), (neg_prob.sigmoid()).cpu().detach().numpy()])
        true_label = np.concatenate([np.ones(h_dst.shape[0]), np.zeros(h_neg.shape[0])])

        ap.append(average_precision_score(true_label, pred_score))
        auc.append(roc_auc_score(true_label, pred_score))
        f1_micro.append(f1_score(true_label, np.where(pred_score > 0.5, 1, 0), average='micro'))
        f1_macro.append(f1_score(true_label, np.where(pred_score > 0.5, 1, 0), average='macro'))

    return {'ap': np.mean(ap), 'auc': np.mean(auc), 'f1_micro': np.mean(f1_micro), 'f1_macro': np.mean(f1_macro)}


def evaluate(g, total_nodes, model, feat, device, infer_data, num_workers=0):
    with torch.no_grad():
        node_emb = model.inference(g, total_nodes, feat, device, 4096, 'cpu')
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

        print('compute metrics...', flush=True)
        
        val_res = compute_metrics(model, node_emb, val_src, val_dst, val_neg, device)
        nn_val_res = compute_metrics(model, node_emb, nn_val_src, nn_val_dst, nn_val_neg, device)
        test_res = compute_metrics(model, node_emb, test_src, test_dst, test_neg, device)
        nn_test_res = compute_metrics(model, node_emb, nn_test_src, nn_test_dst, nn_test_neg, device)

    return val_res, nn_val_res, test_res, nn_test_res


if __name__ == '__main__':
    main()