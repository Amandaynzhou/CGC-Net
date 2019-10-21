import os
from common.utils import mkdirs

class ModelSetting:
    def __init__(self):
        input_dim  =  12
        self.GraphConv_add_self = True
        self.GraphConv_dropout = 0
        self.GraphConv_normalize_embedding = True
        self.GraphConv_input_dim = input_dim
        self.GraphConv_layer_dims = [12, 12]
        self.GraphConv_bias = False
        self.GraphConv_encoder = None
        self.GraphConv_num_class = 3


        self.GcnEncoderGraph_input_dim = input_dim
        self.GcnEncoderGraph_hidden_dim = 32
        self.GcnEncoderGraph_embedding_dim = None
        self.GcnEncoderGraph_label_dim = 3
        self.GcnEncoderGraph_num_layers = None
        self.GcnEncoderGraph_pred_hidden_dims = []
        self.GcnEncoderGraph_concat = True
        self.GcnEncoderGraph_bn = True
        self.GcnEncoderGraph_dropout= 0.0
        self.GcnEncoderGraph_args = None

        self.diffpool_input_dim = input_dim
        self.diffpool_hidden_dim = 32
        self.diffpool_embedding_dim = None
        self.diffpool_label_dim = 3
        self.diffpool_num_layers = None
        self.diffpool_assign_hidden_dim = None
        self.diffpool_assign_ratio = 0.25
        self.diffpool_assign_num_layers = -1,
        self.diffpool_num_pooling = 1
        self.diffpool_pred_hidden_dims = [50]
        self.diffpool_concat = True
        self.diffpool_bn = True
        self.diffpool_dropout = 0.0
        self.diffpool_linkpred = True
        self.diffpool_assign_input_dim = -1
        self.diffpool_args = None

        self.graphsage_num_class = 3


class DataSetting:
    def __init__(self):
        # check in server or not
        self.name = 'colon'
        if 'amanda' in os.getcwd():
            self.root = '/media/amanda/HDD2T_1/warwick-research/data'
            self.save_path = "/media/amanda/HDD2T_1/warwick-research/experiment/gcnn"
            # self.checkpoint = "../../experiment/pretrain/nuclei_seg_weight/best.ckpt-5400"
            self.dataset = ['tialab','extra']
            self.log_path = os.path.join(self.save_path,'log' )
            self.result_path = os.path.join(self.save_path, 'result')
            self.max_num_nodes = 11404 # todo

        else:
            self.root = '/research/dept6/ynzhou/gcnn/data'
            self.save_path = '/research/dept6/ynzhou/gcnn/experiment/gcnn'
            # self.checkpoint = "/mnt/tialab-simon/yanning/experiment/pretrain/new-nuclei-weight/best.ckpt-5400"
            self.log_path = os.path.join(self.save_path,'log' )
            self.result_path = os.path.join(self.save_path, 'result')
            self.dataset = ['tialab','extra']
            self.max_num_nodes = 11404 # 15710
        # self.label  = 'train'
        self.batch_size = 16

        # self.test_data_path_list  = [os.path.join(self.root, 'proto', 'feature', f, self.label) for f in self.dataset]


        # self.avg_num_nodes = 6533
        # self.min_num_nodes = 1643
        self.sample_time = 1
        self.sample_ratio = 0.5
        self.max_edge_distance = 100
        self.max_connected_neighbors = 8
        self.gcnn_type = None
        assert self.gcnn_type in [None, 'baseline']
        self.do_eval = False
        # self.test_data = []
        # self.test_image_list = [os.listdir(f) for f in self.test_data_path_list]
        # test_image_list = []
        # for dataset in self.test_image_list:
        #     namelist = []
        #     for name in dataset:
        #         if 'png' in name:
        #             namelist.append(name)
        #     test_image_list.append(namelist)
        # self.test_image_list = test_image_list
        self.experiment_standard_name = 'N{max_node}_E{max_edge_dist}_{gcnn_method}'.format(max_node = self.max_num_nodes,\
                                                                                            max_edge_dist = self.max_edge_distance,\
                                                                                            gcnn_method = self.gcnn_type)
        # mkdirs(os.path.join(self.save_path, self.dataset, self.label, 'mat'))
        # mkdirs(os.path.join(self.save_path, self.experiment_standard_name, 'log'))
        # with  open(os.path.join(self.save_path, self.experiment_standard_name, 'log')) as f:
        #     for k,v in self.__dict__.items():
        #         f.write(str(k)+':'+str(v))

class CrossValidSetting(DataSetting):
    def __init__(self):
        super(CrossValidSetting, self).__init__()
        self.root = '/research/dept6/ynzhou/gcnn/data'
        self.save_path = '/research/dept6/ynzhou/gcnn/experiment/gcnn-crossval'
        # self.checkpoint = "/mnt/tialab-simon/yanning/experiment/pretrain/new-nuclei-weight/best.ckpt-5400"
        self.log_path = os.path.join(self.save_path,'log' )
        self.result_path = os.path.join(self.save_path, 'result')
        self.dataset = ['shaban']
        self.max_num_nodes = 15615 # 15710
        self.sample_time = 1
        self.max_edge_distance = 100
        self.gcnn_type = None
        assert self.gcnn_type in [None, 'baseline']



class ICIARSetting(DataSetting):
    def __init__(self):
        super(ICIARSetting, self).__init__()
        self.name = 'ICIAR'
        self.dataset = ['ICIAR']
        self.sample_ratio = 1
        self.max_num_nodes =10083

class DataSettingS(DataSetting):
    def __init__(self):
        super(DataSettingS, self).__init__()
        self.max_num_nodes = 15861