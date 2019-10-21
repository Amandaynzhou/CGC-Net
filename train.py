import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from common.metric import ImgLevelResult
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
from torch.multiprocessing import Process
import torch.distributed as dist
# from torch.nn import DataParallel





def evaluate(dataset, model, args, name='Validation', max_num_examples=None):
    model.eval()
    device = 'cuda:1' if torch.cuda.device_count()>1 else 'cuda:0'
    torch.cuda.empty_cache()
    if name != 'train':
        finaleval = ImgLevelResult(args)
    with torch.no_grad():

        test_time = args.test_epoch if (args.dynamic_graph and name !='Train')else 1
        if args.full_test_graph:
            test_time = 1
        pred_n_times = []
        labels_n_time = []# just for check if there is a bug
        for _ in range(test_time):
            preds = []
            labels = []
            dataset.dataset.set_val_epoch(_)
            for batch_idx, data in enumerate(dataset):
                # pdb.set_trace()
                if  args.method in [ 'soft-assign', 'deep-soft-assign', 'soft-assign-jk']:
                    if args.load_data_sparse:
                        for key, item in data:
                            data[key] = data[key].to(device)
                        ypred = model(data)
                        labels.append(data.y)
                    elif (not args.load_data_list) or args.full_test_graph:
                        patch_idx = data['patch_idx']
                        patch_name = dataset.dataset.idxlist[patch_idx.item()]
                        adj = data['adj'].to(device)
                        h0 = data['feats'].to(device)
                        label = data['label']
                        # coor = data['coor']
                        label = label[:, 0].numpy()
                        labels.append(label)
                        batch_num_nodes = data['num_nodes'].cuda()
                        ypred = model((h0, adj, batch_num_nodes))
                        finaleval.patch_result(patch_name, torch.max(ypred, 1)[1].cpu().numpy())
                    else:
                        patch_name = [dataset.dataset.idxlist[d.patch_idx.item()] for d in data]
                        ypred = model(data)
                        label = torch.cat([d.y for d in data]).numpy()
                        labels.append(label)
                        finaleval.batch_patch_result(patch_name, torch.max(ypred, 1)[1].cpu().numpy())
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
                if args.visualization and (batch_idx+1)*args.batch_size<50:
                    if args.method in [ 'soft-assign', 'deep-soft-assign', 'soft-assign-jk']:
                        adj = adj.detach().cpu().numpy()
                        h0 = h0.detach().cpu().numpy()
                        h0 = h0[:, :,-2:]
                        if args.task == 'ICIAR':
                            h0 = h0*[1.,1.3]
                        # h0 = coor.detach().numpy()
                        batch_size = h0.shape[0]
                        name = dataset.dataset.idxlist[int(batch_idx * batch_size) : int((batch_idx+1) * batch_size)]
                        assign_matrix_list = [f.detach().cpu().numpy() for f in model.assign_matrix]
                    else:# for attention pooling
                    # pdb.set_trace()
                        raise NotImplementedError
                        nodes = model.get_perm()
                        nodes = [f.cpu().numpy() for f in nodes]

                    for i in range(adj.shape[0]):
                        # pdb.set_trace()
                        save_name = os.path.join(args.resultdir,gen_prefix(args),'visual', name[i] + '.gexf')
                        assign_matrix_list_single_image = [f[i] for f in assign_matrix_list]
                        output_to_gexf( h0[i], adj[i], assign_matrix_list_single_image, save_name)


            labels = np.concatenate(labels,0)
            preds = np.concatenate(preds,0)
            pred_n_times.append(preds[...,np.newaxis])
            labels_n_time.append(labels[...,np.newaxis])

        if not args.full_test_graph and name !='Train':
            pred_n_times = np.concatenate(pred_n_times, -1)
            labels_n_time = np.hstack(labels_n_time)
            pred_n_times = np.mean(pred_n_times,-1)
            labels_n_time = np.mean(labels_n_time,-1)
            pred_n_times = np.argmax(pred_n_times,1)
        else:
            pred_n_times = pred_n_times[0][...,0]
            labels_n_time = labels_n_time[0][...,0]
            pred_n_times = np.argmax(pred_n_times,1)
    multi_class_acc,binary_acc = finaleval.final_result()
    result = { 'patch_acc': metrics.accuracy_score(labels_n_time,pred_n_times), 'img_acc':multi_class_acc, 'binary_acc': binary_acc }
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
    elif args.method == 'soft-assign-jk':
        name += '_l' + str(args.num_gc_layers) + 'x' + str(args.num_pool)
        name += '_ar' + str(int(args.assign_ratio*100))
        if args.linkpred:
            name += '_lp'
        if args.balanced_cluster:
            name += '_bl' +str(float(args.sita))

    elif args.method == 'deep-soft-assign':
        name += 'p_500-125-30' + str(int(args.assign_ratio * 100))
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
    if args.load_data_sparse:
        name += '_sp'
    if args.sampling_by_number:
        name+= '_sbn'
    if args.load_data_list:
        name +='_list'
    if args.norm_adj:
        name+='_adj0.4'
    if args.activation !='relu':
        name+=args.activation
    if args.readout =='mix':
        name+=args.readout
    if args.task != 'colon':
        name+=('_'+args.task)
    if args.mask !='cia':
        name +='hvnet'
    if args.neighbour !=8:
        name +='_n'+str(args.neighbour)
    name += '_sr' + str(args.sample_ratio)
    if args.drop_out >0:
        name +='_d' + str(args.drop_out)
    if args.add_noise:
        name += '_noise'
    if args.distance_prob_graph:
        name +='d_graph'
    if args.jump_knowledge:
        name +='_jk'
    name += args.graph_sampler
    if args.cross_val!=1:
        name +='_cv'+str(args.cross_val)
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

    device = 'cuda:1' if torch.cuda.device_count()>1 else 'cuda:0'
    start_epoch = 0
    optimizer = init_optim(args.optim, model.parameters(), args.lr, args.weight_decay)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    # optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr= args.lr)
    if args.step_size > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    cudnn.benchmark = True
    iter = 0

    best_val_result = {
            'epoch': 0,
            'loss': 0,
            'img_acc': 0,
            'patch_acc': 0 }
    test_result = {
            'epoch': 0,
            'loss': 0,
            'img_acc': 0,
            'patch_acc':0}
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
    train_iter =  0
    for epoch in range(start_epoch, args.num_epochs):
        if epoch>150:
            break
        torch.cuda.empty_cache()
        total_time = 0
        avg_loss = 0.0
        avg_link_loss = 0.0
        avg_ent_loss = 0.0
        avg_b_loss = 0.0
        model.train()
        print('Epoch: ', epoch)
        is_best = False
        # set epoch

        dataset.dataset.set_epoch(epoch)
        with tqdm(bar_format='{desc}{postfix}') as tq:
            for batch_idx, data in enumerate(tqdm(dataset)):
                train_iter += 40
                # pdb.set_trace()
                begin_time = time.time()
                # pdb.set_trace()
                if args.method in ['soft-assign','deep-soft-assign', 'soft-assign-jk']:
                    # # debug
                    if args.load_data_sparse:
                        for key, item in data:
                            data[key] = data[key].to(device)
                        _, link_loss, ent_loss, cls_loss, b_loss = model(data)
                    elif not args.load_data_list:
                        #dense input
                        adj = data['adj'].to(device)
                        h0 = data['feats'].to(device)
                        batch_num_nodes = data['num_nodes'].to(device)
                        label = data['label'].to(device)
                        label = label[:, 0]
                        _, link_loss, ent_loss, cls_loss, b_loss = model((h0, adj, batch_num_nodes, label))
                    else:

                        _, link_loss, ent_loss, cls_loss, b_loss = model(data)
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
                # del adj,h0,batch_num_nodes,label
                if train_iter%3500 == 0:
                    val_result = evaluate(val_dataset, model, args, name='Validation')
                    val_accs.append(val_result['patch_acc'])
                    if val_result['img_acc'] > best_val_result['img_acc'] - 1e-7:
                        best_val_result['patch_acc'] = val_result['patch_acc']
                        best_val_result['img_acc'] =  val_result['img_acc']
                        best_val_result['epoch'] = epoch
                        is_best = True
                        print('Time:%f, Train loss:%f, Val patch acc:%f img acc:%f  | Best Val acc:%f in epoch%d'
                          % (total_time,

                             loss.detach().cpu().item(),
                             val_result['img_acc'],
                             best_val_result['patch_acc'],
                             best_val_result['img_acc'],
                             best_val_result['epoch']
                             ))
                        save_checkpoint({'epoch': epoch + 1,
                                         'loss': avg_loss,
                                         'state_dict': model.state_dict() if torch.cuda.device_count() < 2 else model.module.state_dict(),
                                         'optimizer': optimizer.state_dict(),
                                         'val_acc': val_result['img_acc']},
                                        is_best, os.path.join(save_path, 'weight.pth.tar'))
                    model.train()
                if args.method in  ['soft-assign','deep-soft-assign', 'soft-assign-jk']:
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
                    if args.method in ['soft-assign','deep-soft-assign', 'soft-assign-jk']:
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
            if args.linkpred and  args.method in ['soft-assign','deep-soft-assign', 'soft-assign-jk']:
                writer.add_scalar('loss/linkpred_loss', avg_link_loss, epoch)
                writer.add_scalar('loss/ent_loss', avg_ent_loss, epoch)
                writer.add_scalar('loss/b_loss', avg_b_loss,epoch)
        # print('Avg loss: ', avg_loss, '; epoch time: ', total_time)

        # train_result = evaluate(dataset, model, args, name='Train', max_num_examples=100)
        # train_accs.append(train_result['img_acc'])
        train_epochs.append(epoch)
        # if val_dataset is not None:
        #     # pdb.set_trace()
        #     val_result = evaluate(val_dataset, model, args, name='Validation')
        #     val_accs.append(val_result['acc'])
        # if val_result['acc'] > best_val_result['acc'] - 1e-7:
        #     best_val_result['acc'] = val_result['acc']
        #     best_val_result['epoch'] = epoch
        #     best_val_result['loss'] = avg_loss
        #     is_best = True
        # print('Time:%f, Train acc:%f, Train loss:%f, Val acc:%f | Best Val acc:%f in epoch%d'
        #       %(total_time,
        #         train_result['acc'],
        #         avg_loss.detach().cpu().item(),
        #         val_result['acc'],
        #         best_val_result['acc'],
        #         best_val_result['epoch']
        #         ))
        #
        # save_checkpoint({'epoch': epoch+1,
        #                  'loss': avg_loss,
        #                  'state_dict': model.state_dict() if torch.cuda.device_count()<2 else model.module.state_dict(),
        #                  'optimizer': optimizer.state_dict(),
        #                  'val_acc':val_result['acc']},
        #                 is_best, os.path.join(save_path, 'weight.pth.tar'))
        if test_dataset is not None:
            test_result = evaluate(test_dataset, model, args, name='Test')
            test_result['epoch'] = epoch
        if writer is not None:
            # writer.add_scalar('acc/train_acc', train_result['img_acc'], epoch)
            writer.add_scalar('acc/val_acc', val_result['img_acc'], epoch)
            writer.add_scalar('loss/best_val_loss', best_val_result['loss'], epoch)
            if test_dataset is not None:
                writer.add_scalar('acc/test_acc', test_result['img_acc'], epoch)

        # print('Best val result: ', best_val_result)
        best_val_epochs.append(best_val_result['epoch'])
        best_val_accs.append(best_val_result['img_acc'])
        if test_dataset is not None:
            print('Test result: ', test_result)
            test_epochs.append(test_result['epoch'])
            test_accs.append(test_result['img_acc'])

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
        if args.task == 'colon':
            args.num_classes = 3
        if args.task == 'ICIAR':
            args.num_classes = 4
        # pdb.set_trace()
        model = network.SoftPoolingGcnEncoder(setting.max_num_nodes,
            input_dim, args.hidden_dim, args.output_dim, True, True, args.hidden_dim,  args.num_classes,
                                              args.assign_ratio,[50],linkpred=args.linkpred, concat= True,
                                              gcn_name= args.gcn_name,collect_assign=args.visualization,
                                              balanced_cluster = args.balanced_cluster,
                                              load_data_sparse=(args.load_data_sparse or args.load_data_list) and not args.full_test_graph,
                                              norm_adj=args.norm_adj, activation=args.activation, readout= args.readout, drop_out=args.drop_out,
                                              jk=args.jump_knowledge,
                                              )
    elif args.method =='deep-soft-assign':
        model = network.DeeperSoftPoolingGcnEncoder(setting.max_num_nodes,
            input_dim, args.hidden_dim, args.output_dim, True, True, args.hidden_dim,  args.num_classes,
                                              args.assign_ratio,[50],linkpred=args.linkpred, concat= True,
                                              gcn_name= args.gcn_name,collect_assign=args.visualization,
                                              balanced_cluster = args.balanced_cluster,
                                              load_data_sparse=(args.load_data_sparse or args.load_data_list)and not args.full_test_graph,
                                                    norm_adj=args.norm_adj, activation=args.activation, readout= args.readout, drop_out=args.drop_out)

    elif args.method =='soft-assign-jk':
        model = network.SoftPoolingGcnEncoderJK(setting.max_num_nodes,
            input_dim, args.hidden_dim, args.output_dim, True, True, args.hidden_dim,  args.num_classes,
                                              args.assign_ratio,[50],linkpred=args.linkpred, concat= True,
                                              gcn_name= args.gcn_name,collect_assign=args.visualization,
                                              balanced_cluster = args.balanced_cluster,
                                              load_data_sparse=(args.load_data_sparse or args.load_data_list) and not args.full_test_graph,
                                              norm_adj=args.norm_adj, activation=args.activation, readout= args.readout, drop_out=args.drop_out,
                                              jk=args.jump_knowledge,
                                              )
    elif args.method == 'soft-assign-one-pool':
        model = network.SoftOnePoolingGcnEncoder(setting.max_num_nodes,
            input_dim, args.hidden_dim, args.output_dim, True, True, args.hidden_dim,  args.num_classes,
                                              args.assign_ratio,[50],linkpred=args.linkpred, concat= True,gcn_name= args.gcn_name)
    elif args.method == 'self-attention-pool':
        model = network.SelfAttentionGraphNetwork(input_dim,args.hidden_dim, args.output_dim, args.bias, args.bn, args.num_classes
                                                 ,args.assign_ratio, [50], name = args.gcn_name )

    elif args.method =='deep-self-attention-pool':
        model = network.DeepSelfAttentionGraphNetwork(input_dim,args.hidden_dim, args.output_dim, args.bias, args.bn, args.num_classes
                                                 ,args.assign_ratio, [50], name = args.gcn_name,  )
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
            resume_path = os.path.join(args.resultdir, gen_prefix(args), resume_file)
        elif args.resume == 'weight':
            resume_file = 'weight.pth.tar'
            resume_path = os.path.join(args.resultdir, gen_prefix(args), resume_file)
        else:
            resume_path  =  os.path.join(args.resultdir,args.resume,'model_best.pth.tar')
        # pdb.set_trace()

        checkpoint = load_checkpoint(resume_path)
        model.load_state_dict(checkpoint['state_dict'])
    # pdb.set_trace()
    if torch.cuda.device_count() > 1 :
        print('use %d GPUs for training!'% torch.cuda.device_count())
        # sync bn
        if args.sync_bn:
            model = convert_model(model)
        if args.load_data_list and not args.full_test_graph:
            model = DataParallel(model).cuda()
        else:
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
    # pdb.set_trace()
    # _ = evaluate(val_loader, model, args, name='train', max_num_examples=30)

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
    _ = evaluate(val_loader, model, args, name='Validation', max_num_examples=None)
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
    parser.add_argument('--num-classes', dest='num_classes', type=int,help='Number of label classes')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest='bn', action='store_const',const=False, default=True, help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', type=float,help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const',const=False, default=True,help='Whether to add bias. Default to True.')
    parser.add_argument('--sample-ratio',dest='sample_ratio', default= 1, )
    parser.add_argument('--sample-time', dest='sample_time',default= 1)
    parser.add_argument('--method', dest='method',help='Method. Possible values: base, base-set2set, soft-assign')
    parser.add_argument('--name-suffix', dest='name_suffix', help='suffix added to the output filename')
    parser.add_argument('--input_feature_dim',dest='input_feature_dim', type=int, help='the feature number for each node', default = 8)
    parser.add_argument('--resume', default= False, )
    parser.add_argument('--optim', dest='optimizer', help = 'name for the optimizer, [adam, sgd, rmsprop] ')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--gamma', default=0.1, type=float, help="learning rate decay")
    parser.add_argument('--step_size', default=10, type=int, metavar='N', help='stepsize to decay learning rate (>0 means this is enabled)')
    parser.add_argument('--skip_train',  action='store_const',
            const=True, default=False, help='only do evaluation')
    parser.add_argument('--visualization', action='store_const',const=True, default=False, help='use assignment matrix for visualization')
    parser.add_argument('--alpha', default = 1, help = 'hyper for link loss')
    parser.add_argument('--beta', default=1, help = 'hyper for ent loss')
    parser.add_argument('--normalize', default= False, help ='normalize the adj matrix or not')
    parser.add_argument('--load_data_list', action='store_true', default= False)
    parser.add_argument('--load_data_sparse', action='store_true', default= False)
    parser.add_argument('--name', default='')
    parser.add_argument('--gcn_name', default='SAGE')
    parser.add_argument('--active', dest='activation',default='relu')
    parser.add_argument('--dynamic_graph' ,dest='dynamic_graph', action='store_const', const=True, default=False,)
    parser.add_argument('--sampling_method', default='random',)
    parser.add_argument('--test_epoch', default=5,type= int)
    parser.add_argument('--sita', default=1.,type= float)
    parser.add_argument('--balanced_cluster',dest='balanced_cluster', action='store_const',const=True, default=False,)
    parser.add_argument('--sync_bn',dest='sync_bn', action='store_const',const=True, default=False,)
    parser.add_argument('--sbn',dest='sampling_by_number', action='store_const',  const=True, default=False,)
    parser.add_argument('--full', dest='full_test_graph' , action='store_const',const=True, default=False,)
    parser.add_argument('--norm_adj',action='store_const', const=True, default=False,)
    parser.add_argument('--readout', default='max', type=str)
    parser.add_argument('--task', default= 'colon', type = str)
    parser.add_argument('--mask',default='cia', type=str)
    parser.add_argument('--n',dest='neighbour', default=8, type=int)
    parser.add_argument('--sample_ratio',default=0.5, type=float)

    parser.add_argument('--drop',dest= 'drop_out' ,default=0.,type=float)
    parser.add_argument('--noise',dest='add_noise', action='store_const', const=True, default=False,)
    parser.add_argument('--valid_full', action='store_const',const=True, default=False,)
    parser.add_argument('--dist_g',dest = 'distance_prob_graph',  action='store_const',const=True, default=False,)
    parser.add_argument('--jk', dest='jump_knowledge', action='store_const', const=True, default=False)
    parser.add_argument('--g', dest='graph_sampler', default='knn', type=str)
    parser.add_argument('--cv', dest='cross_val', default=1, type=int)
    parser.set_defaults(datadir= data_setting.root,
                        logdir=data_setting.log_path,
                        resultdir =data_setting.result_path,

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
    torch.backends.cudnn.benchmark = True
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
    if not prog_args.skip_train:
        with open(os.path.join(result_path,'args.txt'), 'w') as f:
            json.dump(prog_args.__dict__, f, indent=2)

    writer = SummaryWriter(log_path)

    nuclei_graph(prog_args, writer=writer, )

    writer.close()

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()

