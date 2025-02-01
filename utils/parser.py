import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run REARM.")
    parser.add_argument('--seed', type=int, default=2025, help='Seed init.')
    parser.add_argument('--model_name', default='REARM', help='Model name.')
    parser.add_argument('--use_gpu', type=bool, default=True, help='enable CUDA training.')
    parser.add_argument('--gpu_id', type=int, default=0, help='The model of the device running the program')
    parser.add_argument('--dataset', nargs='?', default='baby',
                        help='Choose a dataset from {baby, sports, clothing}')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size.')
    parser.add_argument('--eval_batch_size', type=int, default=8192, help='The data size of batch evaluation')
    parser.add_argument('--metrics', type=list, default=["Precision", "Recall", "NDCG"],
                        help='Choose some from {"Precision", "Recall", "NDCG", "MAP"}')
    parser.add_argument('--topk', type=list, default=[10, 20], help='Metrics scale')
    parser.add_argument('--embedding_dim', type=int, default=64, help='Latent dimension 64.')
    parser.add_argument('--num_epoch', type=int, default=2000, help='Epoch number.')
    parser.add_argument('--num_workers', type=int, default=8, help='Workers number.')
    parser.add_argument('--stopping_step', type=int, default=20, help='early stopping strategy.')
    parser.add_argument('--valid_metric', type=str, default="Recall@20", help='valid metric')
    parser.add_argument('--with_tensorboard', action='store_true', default=False, help='with tensorboard analysis ')

    parser.add_argument('--l_r', type=float, default=5e-5, help='Learning rate.')
    parser.add_argument('--learning_rate_scheduler', type=list, default=[1.0, 50], help='learning rate scheduler.')
    parser.add_argument('--reg_weight', type=float, default=5e-4, help='regularization weight.')
    parser.add_argument('--num_layer', type=int, default=4, help='Layer number.')
    parser.add_argument('--s_drop', type=float, default=0.4, help='self_attention_dropout.')
    parser.add_argument('--m_drop', type=float, default=0.6, help='mutual_attention_dropout.')
    parser.add_argument('--cl_tmp', type=float, default=0.6, help='Contrast learning temperature coefficient')
    parser.add_argument('--cl_loss_weight', type=float, default=5e-6, help='contrast loss weight.')
    parser.add_argument('--diff_loss_weight', type=float, default=1e-4, help='Structure contrast loss weight.')
    parser.add_argument('--user_knn_k', type=int, default=40,
                        help='Select the 10 users most similar to the target users to build the users graph')
    parser.add_argument('--item_knn_k', type=int, default=10,
                        help='Select the 10 items most similar to the target item to build the item graph')

    parser.add_argument('--i_mm_image_weight', type=float, default=0,
                        help='The proportion of visual feat in item graph.')
    parser.add_argument('--u_mm_image_weight', type=float, default=0.2,
                        help='The proportion of visual feat in user graph.')
    parser.add_argument('--n_ii_layers', type=int, default=1,
                        help='Number of layers of item feature propagation in the item graph')
    parser.add_argument('--n_uu_layers', type=int, default=1,
                        help='Number of layers of user feature propagation in the user graph')
    parser.add_argument('--user_aggr_mode', type=str, default='softmax',
                        help='Choose a modedataset from {softmax, mean}')

    parser.add_argument('--rank', type=int, default=3, help='the dimension of low rank matrix decomposition')
    parser.add_argument('--uu_co_weight', type=float, default=0.4,
                        help='the proportion of user co-occurrence graphs to user homographs')
    parser.add_argument('--ii_co_weight', type=float, default=0.2,
                        help='the proportion of item co-occurrence graphs to user homographs')
    return parser.parse_args()
