import os
import torch
import random
import datetime
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from time import time
from collections import defaultdict
from scipy.sparse import coo_matrix
from torch.nn.functional import cosine_similarity


def update_dict(key_ui, dataset, edge_dict):
    for edge in dataset:
        user, item = edge
        edge_dict[user].add(item) if key_ui == "user" else None
        edge_dict[item].add(user) if key_ui == "item" else None
    return edge_dict


def get_local_time():
    return datetime.datetime.now().strftime('%b-%d-%Y-%H-%M-%S')


def cal_reg_loss(cal_embedding):
    return (cal_embedding.norm(2).pow(2)) / cal_embedding.size()[0]


def cal_cos_loss(user, item):
    return 1 - cosine_similarity(user, item, dim=-1).mean()


def init_seed(seed):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def early_stopping(value, best, cur_step, max_step):
    stop_flag = False
    update_flag = False

    if value > best:
        cur_step = 0
        best = value
        update_flag = True
    else:
        cur_step += 1
        if cur_step > max_step:
            stop_flag = True

    return best, cur_step, stop_flag, update_flag


def dict2str(result_dict):
    result_str = ''
    for metric, value in result_dict.items():
        result_str += str(metric) + ': ' + '%.04f' % value + '    '
    return result_str


def res_output(epoch_idx, s_time, e_time, res, t_or_v):
    _output = '\n epoch %d ' % epoch_idx + t_or_v + 'ing [time: %.2fs], ' % (e_time - s_time)
    if t_or_v == "train":
        _output += 'total_loss: {:.4f}, bpr_loss: {:.4f}, reg_loss:{:.4f}, mul_vt_cl_loss: {:.4f}, diff_loss: {:.4f}'.format(
            res[0], res[1], res[2], res[3], res[4])
    elif t_or_v == "valid":
        _output += ' valid result: \n' + dict2str(res)
    else:
        _output += ' test result: \n' + dict2str(res)
    return _output


def get_parameter_number(self):
    self.logger.info(self.model)
    # Print the number of model parameters
    total_num = sum(p.numel() for p in self.model.parameters())
    trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    self.logger.info('Total parameters: {}, Trainable parameters: {}'.format(total_num, trainable_num))


def get_norm_adj_mat(self, interaction_matrix):
    adj_size = (self.n_users + self.n_items, self.n_users + self.n_items)
    A = sp.dok_matrix(adj_size, dtype=np.float32)
    inter_M = interaction_matrix
    inter_M_t = interaction_matrix.transpose()
    # {(userID,itemID):1,(userID,itemID):1}
    data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
    data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                              [1] * inter_M_t.nnz)))
    A._update(data_dict)  # Update to (n_users+n_items)*(n_users+n_items) sparse matrix
    adj = sparse_mx_to_torch_sparse_tensor(A).to(self.device)
    return torch_sparse_tensor_norm_adj(adj, adj, adj_size, self.device)


def cal_sparse_inter_matrix(self, form='coo'):
    src = self.train_dataset[0, :]
    tgt = self.train_dataset[1, :]
    data = np.ones(len(self.train_dataset.transpose(1, 0)))
    mat = coo_matrix((data, (src, tgt)), shape=(self.n_users, self.n_items))

    if form == 'coo':
        return mat
    elif form == 'csr':
        return mat.tocsr()
    else:
        raise NotImplementedError('sparse matrix format [{}] has not been implemented.'.format(form))


def ssl_loss(data1, data2, index, ssl_temp):
    index = torch.unique(index)
    embeddings1 = data1[index]
    embeddings2 = data2[index]
    norm_embeddings1 = F.normalize(embeddings1, p=2, dim=1)
    norm_embeddings2 = F.normalize(embeddings2, p=2, dim=1)
    pos_score_t = torch.sum(torch.mul(norm_embeddings1, norm_embeddings2), dim=1)
    all_score = torch.mm(norm_embeddings1, norm_embeddings2.T)
    pos_score = torch.exp(pos_score_t / ssl_temp)
    all_score = torch.sum(torch.exp(all_score / ssl_temp), dim=1)
    loss = (-torch.sum(torch.log(pos_score / all_score)) / (len(index)))
    return loss


def cal_diff_loss(feat, ui_index, dim):
    """
    :param feat: uv_ut  iv_it (n_users+n_items)*dim*2
    :param ui_index: user or item index
    :return: Squared Frobenius Norm Loss
    """
    input1 = feat[ui_index[:, 0], :dim]
    input2 = feat[ui_index[:, 0], dim:]

    # Zero mean
    input1_mean = torch.mean(input1, dim=0, keepdims=True)
    input2_mean = torch.mean(input2, dim=0, keepdims=True)

    input1 = input1 - input1_mean
    input2 = input2 - input2_mean

    input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
    input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

    input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
    input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

    loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

    return loss


def sele_para(config):
    res = "\n *****************************************************************"
    res += "***************************************************************** \n"
    res += "l_r: " + str(config.learning_rate) + ", reg_w: " + str(config.reg_weight)
    res += ", n_l: " + str(config.n_layers) + ", emb_dim: " + str(config.embedding_dim)
    res += ", s_drop : " + str(config.s_drop) + ", m_drop : " + str(config.m_drop)
    res += ", u_mm_v_w: " + str(config.u_mm_image_weight) + ", i_mm_v_w: " + str(config.i_mm_image_weight)
    res += ", uu_co_w: " + str(config.uu_co_weight) + ", ii_co_w: " + str(config.ii_co_weight)
    res += ", u_knn_k: " + str(config.user_knn_k) + ", i_knn_k: " + str(config.item_knn_k)
    res += ", n_uu_layers: " + str(config.n_uu_layers) + ", n_ii_layers: " + str(config.n_ii_layers)
    res += ", cl_temp: " + str(config.cl_tmp) + ", rank: " + str(config.rank)
    res += ", cl_loss_w: " + str(config.cl_loss_weight) + ", diff_loss_w: " + str(config.diff_loss_weight)
    return res + "\n"


def update_result(self, test_result):
    update_output = ' 🏃 🏃 🏃 🏆🏆🏆 🏃 🏃 🏃      ' + self.model_name + "_" + self.dataset_name + '--Best validation results updated!!!'
    self.logger.info(update_output)
    self.best_test_upon_valid = test_result


def stop_log(self, epoch_idx, run_start_time):
    stop_output = 'Finished training, best eval result in epoch %d' % (epoch_idx - self.cur_step)
    stop_output += "\n [total time: %.2fmins], " % ((time() - run_start_time) / 60)
    stop_output += '\n ' + str(self.config.dataset) + '  key parameter: ' + sele_para(self.config)
    stop_output += 'test result: \n' + dict2str(self.best_test_upon_valid)
    self.logger.info(stop_output)


def plot_curve(self, show=True, save_path=None):
    epochs = list(self.train_loss_dict.keys())
    epochs.sort()
    train_loss_values = [float(self.train_loss_dict[epoch]) for epoch in epochs]
    valid_result_values = [float(self.best_valid_result[epoch]) for epoch in epochs]
    plt.plot(epochs, train_loss_values, label='train', color='red')
    plt.plot(epochs, valid_result_values, label='valid', color='black')
    plt.xticks(epochs)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training loss and Validing result curves')
    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path)


# Generated user or item co-occurrence matrix
def creat_co_occur_matrix(type_ui, all_edge, start_ui, num_ui):
    """
    :param type_ui: Types of created co-occurrence graphs, {user, item}
    :param all_edge: train data np.array([[0, 6], [0, 11], [0, 8], [1, 7]])
    :param start_ui: Minimum or starting user or item index
    :param num_ui:Total number of users or items
    :return:Generated user or item co-occurrence matrix
    """
    edge_dict = defaultdict(set)

    for edge in all_edge:
        user, item = edge
        edge_dict[user].add(item) if type_ui == "user" else None
        edge_dict[item].add(user) if type_ui == "item" else None

    co_graph_matrix = torch.zeros(num_ui, num_ui)
    key_list = sorted(list(edge_dict.keys()))
    bar = tqdm(total=len(key_list))
    for head in range(len(key_list)):
        bar.update(1)
        for rear in range(head + 1, len(key_list)):
            head_key = key_list[head]
            rear_key = key_list[rear]
            ui_head = edge_dict[head_key]
            ui_rear = edge_dict[rear_key]
            inter_len = len(ui_head.intersection(ui_rear))
            if inter_len > 0:
                co_graph_matrix[head_key - start_ui][rear_key - start_ui] = inter_len
                co_graph_matrix[rear_key - start_ui][head_key - start_ui] = inter_len
    bar.close()
    return co_graph_matrix


def creat_dict_graph(co_graph_matrix, num_ui):
    dict_graph = {}
    for i in tqdm(range(num_ui)):
        num_co_ui = len(torch.nonzero(co_graph_matrix[i]))

        if num_co_ui <= 200:
            topk_ui = torch.topk(co_graph_matrix[i], num_co_ui)
            edge_list_i = topk_ui.indices.tolist()
            edge_list_j = topk_ui.values.tolist()
            edge_list = [edge_list_i, edge_list_j]
            dict_graph[i] = edge_list
        else:
            topk_ui = torch.topk(co_graph_matrix[i], 200)
            edge_list_i = topk_ui.indices.tolist()
            edge_list_j = topk_ui.values.tolist()
            edge_list = [edge_list_i, edge_list_j]
            dict_graph[i] = edge_list
    return dict_graph


# Calculate item similarity, build similarity matrix
def get_knn_adj_mat(mm_embeddings, knn_k, device):
    # Standardize and calculate similarity
    context_norm = F.normalize(mm_embeddings, dim=1)
    final_sim = torch.mm(context_norm, context_norm.transpose(1, 0)).cpu()
    sim_value, knn_ind = torch.topk(final_sim, knn_k, dim=-1)
    adj_size = final_sim.size()
    # Construct sparse adjacency matrices
    indices0 = torch.arange(knn_ind.shape[0])
    indices0 = torch.unsqueeze(indices0, 1)
    indices0 = indices0.expand(-1, knn_k)
    indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
    sim_adj = torch.sparse.FloatTensor(indices, sim_value.flatten(), adj_size).to(device)
    degree_adj = torch.sparse.FloatTensor(indices, torch.ones(indices.shape[1]), adj_size)
    return torch_sparse_tensor_norm_adj(sim_adj, degree_adj, adj_size, device)


def torch_sparse_tensor_norm_adj(sim_adj, degree_adj, adj_size, device):
    """
    :param sim_adj: Tensor adjacency matrix (The value of 0 or 1 is degree normalised; the value of [0,1] is similarity normalised)
    :param degree_adj: Tensor adjacency matrix (The value of 0 or 1 is degree normalised; the value of [0,1] is similarity normalised)
    :param adj_size: Tensor size of adjacency matrix
    :param device: cpu or gpu
    :return: Laplace degree normalised adjacency matrix
    """
    # norm adj matrix,add epsilon to avoid Devide by zero Warning
    row_sum = 1e-7 + torch.sparse.sum(degree_adj, -1).to_dense()
    r_inv_sqrt = torch.pow(row_sum, -0.5)

    col = torch.arange(adj_size[0])
    row = torch.arange(adj_size[1])
    sp_degree = torch.sparse.FloatTensor(torch.stack((col, row)).to(device), r_inv_sqrt.to(device))
    return torch.spmm((torch.spmm(sp_degree, sim_adj)), sp_degree)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    if type(sparse_mx) != sp.coo_matrix:
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data).float()
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def topk_sample(n_ui, dict_graph, k, topK_ui, topK_ui_counts, aggr_mode, device):
    ui_graph_index = []
    user_weight_matrix = torch.zeros(len(dict_graph), k)
    for i in range(len(dict_graph)):

        if len(dict_graph[i][0]) < k:
            if len(dict_graph[i][0]) != 0:

                ui_graph_sample = dict_graph[i][0][:k]
                ui_graph_weight = dict_graph[i][1][:k]
                rand_index = np.random.randint(0, len(ui_graph_sample), size=k - len(ui_graph_sample))
                ui_graph_sample += np.array(ui_graph_sample)[rand_index].tolist()
                ui_graph_weight += np.array(ui_graph_weight)[rand_index].tolist()
                ui_graph_index.append(ui_graph_sample)
            else:
                ui_graph_index.append(topK_ui[:k])
                ui_graph_weight = (np.array(topK_ui_counts[:k]) / sum(topK_ui_counts[:k])).tolist()
        else:
            ui_graph_sample = dict_graph[i][0][:k]
            ui_graph_weight = dict_graph[i][1][:k]
            ui_graph_index.append(ui_graph_sample)

        if aggr_mode == 'softmax':
            user_weight_matrix[i] = F.softmax(torch.tensor(ui_graph_weight), dim=0)  # softmax
        elif aggr_mode == 'mean':
            user_weight_matrix[i] = torch.ones(k) / k  # mean

    tmp_all_row = []
    tmp_all_col = []
    for i in range(n_ui):
        row = torch.zeros(1, k) + i
        tmp_all_row += row.flatten()
        tmp_all_col += ui_graph_index[i]
    tmp_all_row = torch.tensor(tmp_all_row).to(torch.int32)
    tmp_all_col = torch.tensor(tmp_all_col).to(torch.int32)
    values = user_weight_matrix.flatten().to(device)
    indices = torch.stack((tmp_all_row, tmp_all_col)).to(device)
    return torch.sparse_coo_tensor(indices, values, (n_ui, n_ui))


def load_or_create_matrix(logger, matrix_type, des, dataset_name, file_name, create_function, *create_args):
    """
    Load a matrix from file if it exists; otherwise, create and save it.
    :param logger: logger
    :param matrix_type: str, type of the matrix (e.g., 'user', 'item').
    :param des: str name of the matrix
    :param dataset_name: str, dataset name used to define the file path.
    :param file_name: str, name of the file to save or load the matrix.
    :param create_function: function, function to call for matrix creation.
    :param create_args: tuple, additional arguments for the create function.
    :return: The loaded or created matrix.
    """
    file_path = os.path.join("data", dataset_name, file_name + ".pt")

    if os.path.exists(file_path):
        matrix = torch.load(file_path)
        logger.info(f"{matrix_type.capitalize()} " + des + " has been loaded!")
    else:
        logger.info(f"{matrix_type.capitalize()} " + des + " does not exist, creating!")
        matrix = create_function(*create_args)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save(matrix, file_path)
        logger.info(f"{matrix_type.capitalize()} " + des + " has been created and saved!")
    return matrix


def propgt_info(ego_feat, n_layers, sp_mat, last_layer=False):
    all_feat = [ego_feat]
    for _ in range(n_layers):
        ego_feat = torch.sparse.mm(sp_mat, ego_feat)
        all_feat += [ego_feat]
    if last_layer:
        return ego_feat

    all_feat = torch.stack(all_feat, dim=1)
    all_feat = all_feat.mean(dim=1, keepdim=False)
    return all_feat
