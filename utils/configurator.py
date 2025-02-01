import os
import torch
import multiprocessing
from utils.helper import init_seed


class Config(object):
    def __init__(self, args):
        self.model_name = args.model_name
        self.dataset = args.dataset
        self.learning_rate = args.l_r
        self.learning_rate_scheduler = args.learning_rate_scheduler
        self.embedding_dim = args.embedding_dim
        self.num_epoch = args.num_epoch
        self.reg_weight = args.reg_weight
        self.use_gpu = args.use_gpu
        self.gpu_id = args.gpu_id
        self.seed = args.seed

        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
        self.topk = args.topk
        self.valid_metric = args.valid_metric
        self.metrics = args.metrics
        self.stopping_step = args.stopping_step
        self.n_layers = args.num_layer

        self.rank = args.rank
        self.s_drop = args.s_drop
        self.m_drop = args.m_drop
        self.cl_tmp = args.cl_tmp
        self.item_knn_k = args.item_knn_k
        self.user_knn_k = args.user_knn_k
        self.num_user_co = args.user_knn_k  # same as user_knn_k to compute
        self.num_item_co = args.item_knn_k  # same as item_knn_k to compute
        self.n_ii_layers = args.n_ii_layers
        self.n_uu_layers = args.n_uu_layers
        self.writer = args.with_tensorboard
        self.uu_co_weight = args.uu_co_weight
        self.ii_co_weight = args.ii_co_weight
        self.cl_loss_weight = args.cl_loss_weight
        self.user_aggr_mode = args.user_aggr_mode
        self.i_mm_image_weight = args.i_mm_image_weight
        self.u_mm_image_weight = args.u_mm_image_weight
        self.diff_loss_weight = args.diff_loss_weight

        self._init_device(args)
        init_seed(self.seed)

    def _init_device(self, args):
        if self.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.use_gpu else "cpu")

        # Ensure that setting up multiple threads does not exceed
        max_cpu_count = multiprocessing.cpu_count()
        self.num_workers = max_cpu_count // 2 if max_cpu_count // 2 < args.num_workers else args.num_workers

    def __str__(self):
        args_info = '\nModel arguments: '
        args_info += ',\n'.join(["{} = {}".format(arg, value) for arg, value in self.__dict__.items()])
        args_info += '.\n'
        return args_info
