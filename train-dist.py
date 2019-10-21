import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import networkx as nx
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from torch.autograd import Variable
import tensorboardX
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import pdb
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import os
import json
import pickle
import random
import shutil
import time
from common.utils import mkdirs, save_checkpoint, load_checkpoint, init_optim, output_to_gexf
# import cross_val
from torch.optim import lr_scheduler
# for debug, i change back to the original encoder to check
from model import encoder_old as encoders
from model import network
from model import diffpool
import gen.feat as featgen
import gen.data as datagen
import cross_val
from graph_sampler import  GraphSampler
# from model import encoders
import util
from torch_geometric.data import DataListLoader
from torch_geometric.nn import DataParallel
import load_data
from dataflow.data import prepare_train_val_loader, DataSetting
from sync_batchnorm import convert_model,patch_replication_callback
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
start_epoch = 0
best_val_result = {
    'epoch': 0,
    'loss': 0,
    'acc': 0}
test_result = {
    'epoch': 0,
    'loss': 0,
    'acc': 0}

def evaluate(dataset, model, args, name='Validation', max_num_examples=None):
    model.eval()


    with torch.no_grad():
        test_time = args.test_epoch if (args.dynamic_graph and name !='Train')else 1
        pred_n_times = []
        labels_n_time = []# just for check if there is a bug
        for _ in range(test_time):
            preds = []
            labels = []
            for batch_idx, data in enumerate(dataset):
                if  args.method == 'soft-assign':
                    adj = data['adj'].cuda()
                    h0 = data['feats'].cuda()
                    label = data['label']
                    label = label[:, 0].numpy()
                    labels.append(label)
                    batch_num_nodes = data['num_nodes'].cuda()
                    ypred = model(h0, adj, batch_num_nodes,)
                else:
                    labels.append(data.y.numpy())
                    data = data.to('cuda')
                    ypred = model(data)
                _, indices = torch.max(ypred, 1)
                preds.append(ypred.detach().cpu().numpy())
                #preds.append(indices.cpu().data.numpy())
                if max_num_examples is not None:
                    if (batch_idx+1)*args.batch_size > max_num_examples:
                        break
                if args.visualization:
                    if args.method == 'soft-assign':
                        adj = adj.detach().cpu().numpy()
                        h0 = h0.detach().cpu().numpy()
                        h0 = h0[:, :,-2:]
                        batch_size = h0.shape[0]
                        name = dataset.dataset.idxlist[int(batch_idx * batch_size) : int((batch_idx+1) * batch_size)]
                        assign_matrix_list = [f.detach().cpu().numpy() for f in model.assign_tensor_list]
                    else:# for attention pooling
                    # pdb.set_trace()
                        nodes = model.get_perm()
                        nodes = [f.cpu().numpy() for f in nodes]

                    for i in range(adj.shape[0]):
                        save_name = os.path.join(args.resultdir, gen_prefix(args),'visual', name[i] + '.gexf')
                        assign_matrix_list_single_image = [f[i] for f in assign_matrix_list]
                        output_to_gexf( h0[i], adj[i], assign_matrix_list_single_image, save_name)


            labels = np.concatenate(labels,0)
            preds = np.concatenate(preds,0)
            pred_n_times.append(preds[...,np.newaxis])
            labels_n_time.append(labels[...,np.newaxis])

        if args.dynamic_graph and name !='Train':
            pred_n_times = np.concatenate(pred_n_times, -1)
            labels_n_time = np.hstack(labels_n_time)
            pred_n_times = np.mean(pred_n_times,-1)
            labels_n_time = np.mean(labels_n_time,-1)
            pred_n_times = np.argmax(pred_n_times,1)
        else:
            pred_n_times = pred_n_times[0][...,0]
            labels_n_time = labels_n_time[0][...,0]
            pred_n_times = np.argmax(pred_n_times,1)

    result = { 'acc': metrics.accuracy_score(labels_n_time,pred_n_times)}
    # print(name, " accuracy:", result['acc'])
    return result

def gen_prefix(args):
    if args.bmname is not None:
        name = args.bmname
    else:
        name = args.dataset
    name += '_' + args.method
    if args.method == 'soft-assign':
        name += '_l' + str(args.num_gc_layers) + 'x' + str(args.num_pool)
        name += '_ar' + str(int(args.assign_ratio*100))
        if args.linkpred:
            name += '_lp'
        if args.balanced_cluster:
            name += '_bl' +str(float(args.sita))
    else:
        name += '_l' + str(args.num_gc_layers)
        name += '_ar' + str(int(args.assign_ratio * 100))
        name += '_' + args.gcn_name
    name += '_h' + str(args.hidden_dim) + '_o' + str(args.output_dim)
    if not args.bias:
        name += '_nobias'
    if len(args.name_suffix) > 0:
        name += '_' + args.name_suffix
    name +=  '_f' +args.feature_type
    name += '_%' + str(args.sample_ratio)
    name += '_a' + str(float(args.alpha) *100)
    name +='_b' + str(float(args.beta) *100)
    # name += '_' + args.sample_method
    name +=args.name
    if args.sync_bn:
        name += '_sync'

    return name

def gen_train_plt_name(args):
    return 'results/' + gen_prefix(args) + '.png'

def log_assignment(assign_tensor, writer, epoch, batch_idx):
    plt.switch_backend('agg')
    fig = plt.figure(figsize=(8,6), dpi=300)

    # has to be smaller than args.batch_size
    for i in range(len(batch_idx)):
        plt.subplot(2, 2, i+1)
        plt.imshow(assign_tensor.cpu().data.numpy()[batch_idx[i]], cmap=plt.get_cmap('BuPu'))
        cbar = plt.colorbar()
        cbar.solids.set_edgecolor("face")
    plt.tight_layout()
    fig.canvas.draw()
    data = tensorboardX.utils.figure_to_image(fig)
    writer.add_image('assignment', data, epoch)

def log_graph(adj, batch_num_nodes, writer, epoch, batch_idx, assign_tensor=None):
    plt.switch_backend('agg')
    fig = plt.figure(figsize=(8,6), dpi=300)

    for i in range(len(batch_idx)):
        ax = plt.subplot(2, 2, i+1)
        num_nodes = batch_num_nodes[batch_idx[i]]
        adj_matrix = adj[batch_idx[i], :num_nodes, :num_nodes].cpu().data.numpy()
        G = nx.from_numpy_matrix(adj_matrix)
        nx.draw(G, pos=nx.spring_layout(G), with_labels=True, node_color='#336699',
                edge_color='grey', width=0.5, node_size=300,
                alpha=0.7)
        ax.xaxis.set_visible(False)

    plt.tight_layout()
    fig.canvas.draw()

    #data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    #data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = tensorboardX.utils.figure_to_image(fig)
    writer.add_image('graphs', data, epoch)

    # log a label-less version
    #fig = plt.figure(figsize=(8,6), dpi=300)
    #for i in range(len(batch_idx)):
    #    ax = plt.subplot(2, 2, i+1)
    #    num_nodes = batch_num_nodes[batch_idx[i]]
    #    adj_matrix = adj[batch_idx[i], :num_nodes, :num_nodes].cpu().data.numpy()
    #    G = nx.from_numpy_matrix(adj_matrix)
    #    nx.draw(G, pos=nx.spring_layout(G), with_labels=False, node_color='#336699',
    #            edge_color='grey', width=0.5, node_size=25,
    #            alpha=0.8)

    #plt.tight_layout()
    #fig.canvas.draw()

    #data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    #data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #writer.add_image('graphs_no_label', data, epoch)

    # colored according to assignment
    assignment = assign_tensor.cpu().data.numpy()
    fig = plt.figure(figsize=(8,6), dpi=300)

    num_clusters = assignment.shape[2]
    all_colors = np.array(range(num_clusters))

    for i in range(len(batch_idx)):
        ax = plt.subplot(2, 2, i+1)
        num_nodes = batch_num_nodes[batch_idx[i]]
        adj_matrix = adj[batch_idx[i], :num_nodes, :num_nodes].cpu().data.numpy()

        label = np.argmax(assignment[batch_idx[i]], axis=1).astype(int)
        label = label[: batch_num_nodes[batch_idx[i]]]
        node_colors = all_colors[label]

        G = nx.from_numpy_matrix(adj_matrix)
        nx.draw(G, pos=nx.spring_layout(G), with_labels=False, node_color=node_colors,
                edge_color='grey', width=0.4, node_size=50, cmap=plt.get_cmap('Set1'),
                vmin=0, vmax=num_clusters-1,
                alpha=0.8)

    plt.tight_layout()
    fig.canvas.draw()

    #data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    #data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = tensorboardX.utils.figure_to_image(fig)
    writer.add_image('graphs_colored', data, epoch)


def train(dataset, model, args, same_feat=True, val_dataset=None, test_dataset=None, writer=None,
        mask_nodes = True, checkpoint = None):
    # pdb.set_trace()
    print("==> Start training")
    writer_batch_idx = [0]

    # start_epoch = 0
    optimizer = init_optim(args.optim, model.parameters(), args.lr, args.weight_decay)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    # optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr= args.lr)
    if args.step_size > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    cudnn.benchmark = True
    iter = 0


    train_accs = []
    train_epochs = []
    best_val_accs = []
    best_val_epochs = []
    test_accs = []
    test_epochs = []
    val_accs = []
    save_path = os.path.join(args.resultdir, gen_prefix(args))
    link_loss = None
    ent_loss = None
    b_loss = None
    # pdb.set_trace()
    for epoch in range(start_epoch, args.num_epochs):
        total_time = 0
        avg_loss = 0.0
        avg_link_loss = 0.0
        avg_ent_loss = 0.0
        avg_b_loss = 0.0
        model.train()
        print('Epoch: ', epoch)
        is_best = False
        with tqdm(bar_format='{desc}{postfix}') as tq:
            for batch_idx, data in enumerate(tqdm(dataset)):
                # pdb.set_trace()
                begin_time = time.time()
                if args.method == 'soft-assign':
                    adj = data['adj'].cuda()
                    h0 = data['feats'].cuda()
                    batch_num_nodes = data['num_nodes'].cuda()
                    label = data['label'].cuda()
                    label = label[:, 0]
                    _, link_loss, ent_loss, cls_loss, b_loss = model(h0, adj, batch_num_nodes, label)
                    link_loss = torch.mean(link_loss)
                    ent_loss = torch.mean(ent_loss)
                    cls_loss = torch.mean(cls_loss)
                    b_loss = torch.mean(b_loss)
                    # cls_loss =  F.nll_loss(F.log_softmax(ypred, -1), label)
                # pdb.set_trace()

                    loss = float(args.alpha) * link_loss + float(args.beta) * ent_loss +  cls_loss + float(args.sita) * b_loss
                else:
                    # attention pool
                    data = data.to('cuda')
                    loss = model(data)
                    # loss = torch.mean(loss)
                optimizer.zero_grad()
                loss.backward()
                # todo : check if clip is necessary.
                # nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                iter += 1
                if args.method == 'soft-assign':
                    avg_loss += loss.detach()
                    avg_ent_loss += ent_loss.detach()
                    avg_link_loss += link_loss.detach()
                    avg_b_loss += b_loss.detach()
                else:
                    avg_loss += loss.detach()
                    avg_ent_loss += 0
                    avg_link_loss += 0
                    avg_b_loss += 0
                elapsed = time.time() - begin_time
                total_time += elapsed
                # log once per XX epochs
                # if epoch % 10 == 0 and batch_idx == len(dataset) // 2 and args.method == 'soft-assign' and writer is not None:
                #     log_assignment(model.assign_tensor, writer, epoch, writer_batch_idx)
                if iter % 2 == 0:
                    tq.set_description('Processing iter=%d'%iter)
                    if args.method == 'soft-assign':
                        tq.set_postfix({'overall':  loss.item(),
                                        'cls': cls_loss.item(),
                                        'plink': link_loss.item(),
                                        'ent': ent_loss.item(),
                                        'b': b_loss.item()},
                                        )
                    else:
                        tq.set_postfix({'overall':  loss.item(),
                                        })
        # decay lr
        if args.step_size > 0: scheduler.step()
        avg_loss /= batch_idx + 1
        avg_link_loss /= batch_idx + 1
        avg_ent_loss /= batch_idx + 1
        avg_b_loss /= batch_idx + 1
        if writer is not None:
            writer.add_scalar('loss/avg_loss', avg_loss, epoch)
            if args.linkpred and  args.method == 'soft-assign':
                writer.add_scalar('loss/linkpred_loss', avg_link_loss, epoch)
                writer.add_scalar('loss/ent_loss', avg_ent_loss, epoch)
                writer.add_scalar('loss/b_loss', avg_b_loss,epoch)
        # print('Avg loss: ', avg_loss, '; epoch time: ', total_time)

        train_result = evaluate(dataset, model, args, name='Train', max_num_examples=100)
        train_accs.append(train_result['acc'])
        train_epochs.append(epoch)
        if val_dataset is not None:
            # pdb.set_trace()
            val_result = evaluate(val_dataset, model, args, name='Validation')
            val_accs.append(val_result['acc'])
        if val_result['acc'] > best_val_result['acc'] - 1e-7:
            best_val_result['acc'] = val_result['acc']
            best_val_result['epoch'] = epoch
            best_val_result['loss'] = avg_loss
            is_best = True
        print('Time:%f, Train acc:%f, Train loss:%f, Val acc:%f | Best Val acc:%f in epoch%d'
              %(total_time,
                train_result['acc'],
                avg_loss.detach().cpu().item(),
                val_result['acc'],
                best_val_result['acc'],
                best_val_result['epoch']
                ))

        save_checkpoint({'epoch': epoch+1,
                         'loss': avg_loss,
                         'state_dict': model.state_dict() if torch.cuda.device_count()<2 else model.module.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'val_acc':val_result['acc']},
                        is_best, os.path.join(save_path, 'weight.pth.tar'))
        if test_dataset is not None:
            test_result = evaluate(test_dataset, model, args, name='Test')
            test_result['epoch'] = epoch
        if writer is not None:
            writer.add_scalar('acc/train_acc', train_result['acc'], epoch)
            writer.add_scalar('acc/val_acc', val_result['acc'], epoch)
            writer.add_scalar('loss/best_val_loss', best_val_result['loss'], epoch)
            if test_dataset is not None:
                writer.add_scalar('acc/test_acc', test_result['acc'], epoch)

        # print('Best val result: ', best_val_result)
        best_val_epochs.append(best_val_result['epoch'])
        best_val_accs.append(best_val_result['acc'])
        if test_dataset is not None:
            print('Test result: ', test_result)
            test_epochs.append(test_result['epoch'])
            test_accs.append(test_result['acc'])

    matplotlib.style.use('seaborn')
    plt.switch_backend('agg')
    plt.figure()
    plt.plot(train_epochs, util.exp_moving_avg(train_accs, 0.85), '-', lw=1)
    if test_dataset is not None:
        plt.plot(best_val_epochs, best_val_accs, 'bo', test_epochs, test_accs, 'go')
        plt.legend(['train', 'val', 'test'])
    else:
        plt.plot(best_val_epochs, best_val_accs, 'bo')
        plt.legend(['train', 'val'])
    plt.savefig(gen_train_plt_name(args), dpi=600)
    plt.close()
    matplotlib.style.use('default')

    return model, val_accs

def prepare_data(graphs, args, test_graphs=None, max_nodes=0):

    random.shuffle(graphs)
    if test_graphs is None:
        train_idx = int(len(graphs) * args.train_ratio)
        test_idx = int(len(graphs) * (1-args.test_ratio))
        train_graphs = graphs[:train_idx]
        val_graphs = graphs[train_idx: test_idx]
        test_graphs = graphs[test_idx:]
    else:
        train_idx = int(len(graphs) * args.train_ratio)
        train_graphs = graphs[:train_idx]
        val_graphs = graphs[train_idx:]
    print('Num training graphs: ', len(train_graphs), 
          '; Num validation graphs: ', len(val_graphs),
          '; Num testing graphs: ', len(test_graphs))

    print('Number of graphs: ', len(graphs))
    print('Number of edges: ', sum([G.number_of_edges() for G in graphs]))
    print('Max, avg, std of graph size: ', 
            max([G.number_of_nodes() for G in graphs]), ', '
            "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])), ', '
            "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))

    # minibatch
    dataset_sampler = GraphSampler(train_graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    train_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.num_workers,
            )

    dataset_sampler = GraphSampler(val_graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    val_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers)

    dataset_sampler = GraphSampler(test_graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    test_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers)

    return train_dataset_loader, val_dataset_loader, test_dataset_loader, \
            dataset_sampler.max_num_nodes, dataset_sampler.feat_dim, dataset_sampler.assign_feat_dim

def syn_community1v2(args, writer=None, export_graphs=False):

    # data
    graphs1 = datagen.gen_ba(range(40, 60), range(4, 5), 500, 
            featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float)))
    for G in graphs1:
        G.graph['label'] = 0
    if export_graphs:
        util.draw_graph_list(graphs1[:16], 4, 4, 'figs/ba')

    graphs2 = datagen.gen_2community_ba(range(20, 30), range(4, 5), 500, 0.3, 
            [featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))])
    for G in graphs2:
        G.graph['label'] = 1
    if export_graphs:
        util.draw_graph_list(graphs2[:16], 4, 4, 'figs/ba2')

    graphs = graphs1 + graphs2
    
    train_dataset, val_dataset, test_dataset, max_num_nodes, input_dim, assign_input_dim = prepare_data(graphs, args)
    if args.method == 'soft-assign':
        print('Method: soft-assign')
        model = encoders.SoftPoolingGcnEncoder(
                max_num_nodes, 
                input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
                bn=args.bn, linkpred=args.linkpred, assign_input_dim=assign_input_dim).cuda()
    elif args.method == 'base-set2set':
        print('Method: base-set2set')
        model = encoders.GcnSet2SetEncoder(input_dim, args.hidden_dim, args.output_dim, 2,
                args.num_gc_layers, bn=args.bn).cuda()
    else:
        print('Method: base')
        model = encoders.GcnEncoderGraph(input_dim, args.hidden_dim, args.output_dim, 2,
                args.num_gc_layers, bn=args.bn).cuda()

    train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=test_dataset,
            writer=writer)

def syn_community2hier(args, writer=None):

    # data
    feat_gen = [featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))]
    graphs1 = datagen.gen_2hier(1000, [2,4], 10, range(4,5), 0.1, 0.03, feat_gen)
    graphs2 = datagen.gen_2hier(1000, [3,3], 10, range(4,5), 0.1, 0.03, feat_gen)
    graphs3 = datagen.gen_2community_ba(range(28, 33), range(4,7), 1000, 0.25, feat_gen)

    for G in graphs1:
        G.graph['label'] = 0
    for G in graphs2:
        G.graph['label'] = 1
    for G in graphs3:
        G.graph['label'] = 2

    graphs = graphs1 + graphs2 + graphs3

    train_dataset, val_dataset, test_dataset, max_num_nodes, input_dim, assign_input_dim = prepare_data(graphs, args)

    if args.method == 'soft-assign':
        print('Method: soft-assign')
        model = encoders.SoftPoolingGcnEncoder(
                max_num_nodes, 
                input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
                bn=args.bn, linkpred=args.linkpred, args=args, assign_input_dim=assign_input_dim).cuda()
    elif args.method == 'base-set2set':
        print('Method: base-set2set')
        model = encoders.GcnSet2SetEncoder(input_dim, args.hidden_dim, args.output_dim, 2,
                args.num_gc_layers, bn=args.bn, args=args, assign_input_dim=assign_input_dim).cuda()
    else:
        print('Method: base')
        model = encoders.GcnEncoderGraph(input_dim, args.hidden_dim, args.output_dim, 2,
                args.num_gc_layers, bn=args.bn, args=args).cuda()
    train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=test_dataset,
            writer=writer)


def pkl_task(args, feat=None):
    with open(os.path.join(args.datadir, args.pkl_fname), 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    graphs = data[0]
    labels = data[1]
    test_graphs = data[2]
    test_labels = data[3]

    for i in range(len(graphs)):
        graphs[i].graph['label'] = labels[i]
    for i in range(len(test_graphs)):
        test_graphs[i].graph['label'] = test_labels[i]

    if feat is None:
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)
        for G in test_graphs:
            featgen_const.gen_node_features(G)

    train_dataset, test_dataset, max_num_nodes = prepare_data(graphs, args, test_graphs=test_graphs)
    model = encoders.GcnEncoderGraph(
            args.input_dim, args.hidden_dim, args.output_dim, args.num_classes, 
            args.num_gc_layers, bn=args.bn).cuda()
    train(train_dataset, model, args, test_dataset=test_dataset)
    evaluate(test_dataset, model, args, 'Validation')

def benchmark_task(args, writer=None, feat='node-label'):
    graphs = load_data.read_graphfile(args.datadir, args.bmname, max_nodes=args.max_nodes)
    
    if feat == 'node-feat' and 'feat_dim' in graphs[0].graph:
        print('Using node features')
        input_dim = graphs[0].graph['feat_dim']
    elif feat == 'node-label' and 'label' in graphs[0].node[0]:
        print('Using node labels')
        for G in graphs:
            for u in G.nodes():
                G.node[u]['feat'] = np.array(G.node[u]['label'])
    else:
        print('Using constant labels')
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)

    train_dataset, val_dataset, test_dataset, max_num_nodes, input_dim, assign_input_dim = \
            prepare_data(graphs, args, max_nodes=args.max_nodes)
    if args.method == 'soft-assign':
        print('Method: soft-assign')
        model = diffpool.SoftPoolingGcnEncoder(
                max_num_nodes, 
                input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
                bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
                assign_input_dim=assign_input_dim).cuda()
    elif args.method == 'base-set2set':
        print('Method: base-set2set')
        model = encoders.GcnSet2SetEncoder(
                input_dim, args.hidden_dim, args.output_dim, args.num_classes,
                args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).cuda()
    else:
        print('Method: base')
        model = encoders.GcnEncoderGraph(
                input_dim, args.hidden_dim, args.output_dim, args.num_classes, 
                args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).cuda()

    train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=test_dataset,
            writer=writer)
    # evaluate(test_dataset, model, args, 'Validation')


def nuclei_graph_main_worker(gpu, ngpus_per_node, args, writer = None,):
    global start_epoch,best_val_result,test_result
    pdb.set_trace()
    args.gpu = gpu
    args.ngpus_per_node = ngpus_per_node
    print("Use GPU: {} for training".format(args.gpu))
    args.rank = args.rank * ngpus_per_node + gpu
    torch.cuda.set_device(args.gpu)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

    cudnn.benchmark = True
    args.batch_size = int(args.batch_size / args.world_size)
    args.num_workers = int(args.num_workers / args.world_size)
    print("batch size is {}, worker is {}".format(args.batch_size, args.num_workers))

    train_loader, val_loader, test_loader = prepare_train_val_loader(args)
    # import pdb;
    # pdb.set_trace()
    setting = DataSetting()
    input_dim = args.input_feature_dim
    assign_input_dim = input_dim
    if args.method == 'soft-assign':
        print('Method: soft-assign')

        model = network.SoftPoolingGcnEncoder(setting.max_num_nodes,
            input_dim, args.hidden_dim, args.output_dim, True, True, args.hidden_dim,  args.num_classes,
                                              args.assign_ratio,[50],linkpred=args.linkpred, concat= True,
                                              gcn_name= args.gcn_name,balanced_cluster = args.balanced_cluster)
    elif args.method == 'soft-assign-one-pool':
        model = network.SoftOnePoolingGcnEncoder(setting.max_num_nodes,
            input_dim, args.hidden_dim, args.output_dim, True, True, args.hidden_dim,  args.num_classes,
                                              args.assign_ratio,[50],linkpred=args.linkpred, concat= True,gcn_name= args.gcn_name)
    elif args.method == 'self-attention-pool':
        model = network.SelfAttentionGraphNetwork(input_dim,args.hidden_dim, args.output_dim, args.bias, args.bn, args.num_classes
                                                 ,args.assign_ratio, [50], name = args.gcn_name )

    elif args.method =='deep-self-attention-pool':
        model = network.DeepSelfAttentionGraphNetwork(input_dim,args.hidden_dim, args.output_dim, args.bias, args.bn, args.num_classes
                                                 ,args.assign_ratio, [50], name = args.gcn_name )
    elif args.method == 'soft-assign-old':
        print('Method: soft-assign-old')
        model = encoders.SoftPoolingGcnEncoder(
            setting.max_num_nodes,
            input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
            args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
            bn=args.bn, dropout=args.dropout, linkpred=False, args=args,
            assign_input_dim=assign_input_dim)
    elif args.method == 'diffpool':
        print('diffpool')
        model = diffpool.SoftPoolingGcnEncoder(setting.max_num_nodes,input_dim,args.hidden_dim,args.output_dim,
                                               args.num_classes, args.num_gc_layers,args.hidden_dim, assign_ratio=args.assign_ratio,
                                               num_pooling=2,bn=args.bn, dropout=args.dropout, linkpred=False, args=args,
            assign_input_dim=assign_input_dim)

    elif args.method == 'base-set2set':
        print('Method: base-set2set')
        model = encoders.GcnSet2SetEncoder(
            input_dim, args.hidden_dim, args.output_dim, args.num_classes,
            args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args)
    else:
        print('Method: base')
        model = encoders.GcnEncoderGraph(
            input_dim, args.hidden_dim, args.output_dim, args.num_classes,
            args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args)
    #resume
    if(args.resume):
        if args.resume == 'best':
            resume_file = 'model_best.pth.tar'
        else:
            resume_file = 'weight.pth.tar'
        # pdb.set_trace()
        resume_path = os.path.join(args.resultdir, gen_prefix(args),resume_file)
        checkpoint = load_checkpoint(resume_path)
        model.load_state_dict(checkpoint['state_dict'])
    # pdb.set_trace()

    # if args.sync_bn:
    #     model = convert_model(model)
    model = model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu,
                                                      )

    # if args.sync_bn:
    #     patch_replication_callback(model)
        # model = DataParallel(model).cuda()
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # model = model.to(device)
        # device = torch.device()
    # pdb.set_trace()

    if not args.skip_train:
        if args.resume:
            _, val_accs = train(train_loader, model, args, val_dataset=val_loader, test_dataset=None,
            writer=writer, checkpoint = checkpoint)
        else:
            _, val_accs = train(train_loader, model, args, val_dataset=val_loader, test_dataset=None,
            writer=writer, )
        print('finally: max_val_acc:%f'%max(val_accs))


    # print('skip training, do evaluation')
    # pdb.set_trace()
    # resume_path = os.path.join(args.resultdir, gen_prefix(args),  'model_best.pth.tar')
    # checkpoint = load_checkpoint(resume_path)['state_dict']
    # model.load_state_dict(checkpoint)
    _ = evaluate(test_loader, model, args, name='Test')
    print(_)


def nuclei_graph(args, writer = None):

    train_loader, val_loader, test_loader = prepare_train_val_loader(args)
    # import pdb;
    # pdb.set_trace()
    setting = DataSetting()
    input_dim = args.input_feature_dim
    assign_input_dim = input_dim
    if args.method == 'soft-assign':
        print('Method: soft-assign')
        # model = encoders.SoftPoolingGcnEncoder(
        #     setting.avg_num_nodes,
        #     input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
        #     args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
        #     bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
        #     assign_input_dim=assign_input_dim)

        model = network.SoftPoolingGcnEncoder(setting.max_num_nodes,
            input_dim, args.hidden_dim, args.output_dim, True, True, args.hidden_dim,  args.num_classes,
                                              args.assign_ratio,[50],linkpred=args.linkpred, concat= True,
                                              gcn_name= args.gcn_name,balanced_cluster = args.balanced_cluster)
    elif args.method == 'soft-assign-one-pool':
        model = network.SoftOnePoolingGcnEncoder(setting.max_num_nodes,
            input_dim, args.hidden_dim, args.output_dim, True, True, args.hidden_dim,  args.num_classes,
                                              args.assign_ratio,[50],linkpred=args.linkpred, concat= True,gcn_name= args.gcn_name)
    elif args.method == 'self-attention-pool':
        model = network.SelfAttentionGraphNetwork(input_dim,args.hidden_dim, args.output_dim, args.bias, args.bn, args.num_classes
                                                 ,args.assign_ratio, [50], name = args.gcn_name )

    elif args.method =='deep-self-attention-pool':
        model = network.DeepSelfAttentionGraphNetwork(input_dim,args.hidden_dim, args.output_dim, args.bias, args.bn, args.num_classes
                                                 ,args.assign_ratio, [50], name = args.gcn_name )
    elif args.method == 'soft-assign-old':
        print('Method: soft-assign-old')
        model = encoders.SoftPoolingGcnEncoder(
            setting.max_num_nodes,
            input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
            args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
            bn=args.bn, dropout=args.dropout, linkpred=False, args=args,
            assign_input_dim=assign_input_dim)
    elif args.method == 'diffpool':
        print('diffpool')
        model = diffpool.SoftPoolingGcnEncoder(setting.max_num_nodes,input_dim,args.hidden_dim,args.output_dim,
                                               args.num_classes, args.num_gc_layers,args.hidden_dim, assign_ratio=args.assign_ratio,
                                               num_pooling=2,bn=args.bn, dropout=args.dropout, linkpred=False, args=args,
            assign_input_dim=assign_input_dim)

    elif args.method == 'base-set2set':
        print('Method: base-set2set')
        model = encoders.GcnSet2SetEncoder(
            input_dim, args.hidden_dim, args.output_dim, args.num_classes,
            args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args)
    else:
        print('Method: base')
        model = encoders.GcnEncoderGraph(
            input_dim, args.hidden_dim, args.output_dim, args.num_classes,
            args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args)
    #resume
    if(args.resume):
        if args.resume == 'best':
            resume_file = 'model_best.pth.tar'
        else:
            resume_file = 'weight.pth.tar'
        # pdb.set_trace()
        resume_path = os.path.join(args.resultdir, gen_prefix(args),resume_file)
        checkpoint = load_checkpoint(resume_path)
        model.load_state_dict(checkpoint['state_dict'])
    # pdb.set_trace()
    if torch.cuda.device_count() > 1 :
        print('use %d GPUs for training!'% torch.cuda.device_count())
        # sync bn
        if args.sync_bn:
            model = convert_model(model)
        model = nn.DataParallel(model).cuda()
        if args.sync_bn:
            patch_replication_callback(model)
        # model = DataParallel(model).cuda()
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # model = model.to(device)
        # device = torch.device()
    # pdb.set_trace()
    else:
        model = model.cuda()

    if not args.skip_train:
        if args.resume:
            _, val_accs = train(train_loader, model, args, val_dataset=val_loader, test_dataset=None,
            writer=writer, checkpoint = checkpoint)
        else:
            _, val_accs = train(train_loader, model, args, val_dataset=val_loader, test_dataset=None,
            writer=writer, )
        print('finally: max_val_acc:%f'%max(val_accs))


    # print('skip training, do evaluation')
    # pdb.set_trace()
    # resume_path = os.path.join(args.resultdir, gen_prefix(args),  'model_best.pth.tar')
    # checkpoint = load_checkpoint(resume_path)['state_dict']
    # model.load_state_dict(checkpoint)
    _ = evaluate(test_loader, model, args, name='Test')
    print(_)

def arg_parse():
    data_setting = DataSetting()

    parser = argparse.ArgumentParser(description='GraphPool arguments.')
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument('--dataset', dest='dataset', 
            help='Input dataset.')
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument('--bmname', dest='bmname',
            help='Name of the benchmark dataset')
    io_parser.add_argument('--pkl', dest='pkl_fname',
            help='Name of the pkl data file')

    softpool_parser = parser.add_argument_group()
    softpool_parser.add_argument('--assign-ratio', dest='assign_ratio', type=float,
            help='ratio of number of nodes in consecutive layers')
    softpool_parser.add_argument('--num-pool', dest='num_pool', type=int,
            help='number of pooling layers')
    parser.add_argument('--linkpred', dest='linkpred', action='store_true')
    parser.add_argument('--datadir', dest='datadir',
            help='Directory where benchmark is located')
    parser.add_argument('--logdir', dest='logdir',
            help='Tensorboard log directory')
    parser.add_argument('--cuda', dest='cuda',
            help='CUDA.')
    parser.add_argument('--max-nodes', dest='max_nodes', type=int,
            help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
    parser.add_argument('--clip', dest='clip', type=float,
            help='Gradient clipping.')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
            help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int,
            help='Number of epochs to train.')
    # parser.add_argument('--train-ratio', dest='train_ratio', type=float,
    #         help='Ratio of number of graphs training set to all graphs.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
            help='Number of workers to load data.')
    parser.add_argument('--feature', dest='feature_type', default='ca',
                        help = '[c, ca, cal, cl] c: coor, a:appearance, l:soft-label')
    parser.add_argument('--input-dim', dest='input_dim', type=int,
            help='Input feature dimension')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int,
            help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', type=int,
            help='Output dimension')
    parser.add_argument('--num-classes', dest='num_classes', type=int,
            help='Number of label classes')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int,
            help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest='bn', action='store_const',
            const=False, default=True,
            help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', type=float,
            help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const',
            const=False, default=True,
            help='Whether to add bias. Default to True.')
    parser.add_argument('--sample-ratio',dest='sample_ratio', default= 1, )
    parser.add_argument('--sample-time', dest='sample_time',default= 1)
    parser.add_argument('--method', dest='method',
            help='Method. Possible values: base, base-set2set, soft-assign')
    parser.add_argument('--name-suffix', dest='name_suffix',
            help='suffix added to the output filename')
    parser.add_argument('--input_feature_dim',dest='input_feature_dim', type=int,
            help='the feature number for each node', default = 8)
    parser.add_argument('--resume', default= False, )
    parser.add_argument('--optim', dest='optimizer', help = 'name for the optimizer, [adam, sgd, rmsprop] ')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--gamma', default=0.1, type=float, help="learning rate decay")
    parser.add_argument('--step_size', default=10, type=int, metavar='N',
                    help='stepsize to decay learning rate (>0 means this is enabled)')
    parser.add_argument('--skip_train',  action='store_const',
            const=True, default=False, help='only do evaluation')
    parser.add_argument('--visualization', default= False, type = bool, help='use assignment matrix for visualization')
    parser.add_argument('--alpha', default = 1, help = 'hyper for link loss')
    parser.add_argument('--beta', default=1, help = 'hyper for ent loss')
    parser.add_argument('--normalize', default= False, help ='normalize the adj matrix or not')
    parser.add_argument('--load_data_list', action='store_true', default= False)
    parser.add_argument('--load_data_sparse', action='store_true', default= False)
    parser.add_argument('--name', default='')
    parser.add_argument('--gcn_name', default='SAGE')
    parser.add_argument('--dynamic_graph' ,dest='dynamic_graph', action='store_const',
            const=True, default=False,)
    parser.add_argument('--sampling_method', default='random',)
    parser.add_argument('--test_epoch', default=5)
    parser.add_argument('--sita', default=1.,type= float)
    parser.add_argument('--balanced_cluster',dest='balanced_cluster', action='store_const',
            const=True, default=False,)
    parser.add_argument('--sync_bn',dest='sync_bn', action='store_const',
            const=True, default=False,)
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://137.189.90.114:2224', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')


    parser.set_defaults(datadir= data_setting.root,
                        logdir=data_setting.log_path,
                        resultdir =data_setting.result_path,
                        sample_ratio = data_setting.sample_ratio,
                        sample_time = data_setting.sample_time,
                        dataset='nuclei',
                        max_nodes=16000, # no use
                        cuda='0',
                        feature='cl',
                        lr=0.001,
                        clip=2.0,
                        batch_size=3,
                        num_epochs=1000,
                        num_workers=4,
                        input_dim=10,
                        hidden_dim=20,
                        output_dim=20,
                        num_classes=3,
                        num_gc_layers=3,
                        dropout=0.0,
                        method='soft-assign',
                        name_suffix='',
                        assign_ratio=0.1,
                        num_pool=1,
                        input_feature_dim = 8,
                        optim = 'adam',
                        weight_decay = 1e-4,
                        step_size = 20,
                        gamma = 0.1,
                        dynamic_graph= False,
                        test_epoch = 5,
                       )
    return parser.parse_args()



def main():
    # pdb.set_trace()
    prog_args = arg_parse()

    # export scalar data to JSON for external processing
    log_path = os.path.join(prog_args.logdir, gen_prefix(prog_args))
    result_path = os.path.join(prog_args.resultdir , gen_prefix(prog_args))
    if prog_args.visualization:
        visual_path = os.path.join(prog_args.resultdir, gen_prefix(prog_args), 'visual')

    mkdirs(log_path)
    mkdirs(result_path)
    if prog_args.visualization:
        mkdirs(visual_path)
    # save argparser
    with open(os.path.join(result_path,'args.txt'), 'w') as f:
        json.dump(prog_args.__dict__, f, indent=2)

    writer = SummaryWriter(log_path)


    if torch.cuda.device_count() > 1:
        prog_args.world_size = torch.cuda.device_count()
        nuclei_graph_main_worker(0,1,prog_args)
        #mp.spawn(nuclei_graph_main_worker, nprocs=1,args=(1,prog_args))
    else:
        nuclei_graph(prog_args, writer=writer)

    writer.close()

if __name__ == "__main__":
    main()

