import math
from utils.helper import *
from logging import getLogger
from collections import defaultdict
import torch.sparse as tsp


class BaseDataset(object):
    def __init__(self, config):
        self.config = config
        self.logger = getLogger("normal")
        self.device = config.device
        self.dataset_name = config.dataset
        self.load_all_data()
        self.processed_eval_data()
        self.n_users = len(set(self.train_data[:, 0]) | set(self.valid_data[:, 0]) | set(self.test_data[:, 0]))
        self.n_items = len(set(self.train_data[:, 1]) | set(self.valid_data[:, 1]) | set(self.test_data[:, 1]))
        self.train_data[:, 1] += self.n_users  # Ensure that the ids are different
        self.valid_data[:, 1] += self.n_users  # Ensure that the ids are different
        self.test_data[:, 1] += self.n_users  # Ensure that the ids are different
        self.dict_user_items()

    def load_all_data(self):
        dataset_path = str('./data/' + self.dataset_name)
        self.train_dataset = np.load(dataset_path + '/train.npy', allow_pickle=True)  # [[1,2,3],[2,3,0]]
        v_feat = np.load(dataset_path + '/image_feat.npy', allow_pickle=True)
        self.i_v_feat = torch.from_numpy(v_feat).type(torch.FloatTensor).to(self.device)  # 4096
        t_feat = np.load(dataset_path + '/text_feat.npy', allow_pickle=True)
        self.i_t_feat = torch.from_numpy(t_feat).type(torch.FloatTensor).to(self.device)  # 384
        self.valid_dataset = np.load(dataset_path + '/valid.npy', allow_pickle=True)  # [[1,2,3],[2,3,0]]
        self.test_dataset = np.load(dataset_path + '/test.npy', allow_pickle=True)  # [[1,2,3],[2,3,0]]

    def processed_eval_data(self):
        self.train_data = self.train_dataset.transpose(1, 0).copy()
        self.valid_data = self.valid_dataset.transpose(1, 0).copy()
        self.test_data = self.test_dataset.transpose(1, 0).copy()

    def load_eval_data(self):
        return self.valid_data, self.test_data

    def dict_user_items(self):
        self.dict_train_u_i = update_dict("user", self.train_data, defaultdict(set))
        self.dict_train_i_u = update_dict("item", self.train_data, defaultdict(set))
        tmp_dict_u_i = update_dict("user", self.valid_data, self.dict_train_u_i)
        self.user_items_dict = update_dict("user", self.test_data, tmp_dict_u_i)

        # Process out the most interacted users
        # (first sort by the number of users interacting with the user, and finally return the user values in descending order)
        sort_itme_num = sorted(self.dict_train_u_i.items(), key=lambda item: len(item[1]), reverse=True)
        self.topK_users = [temp[0] for temp in sort_itme_num]
        self.topK_users_counts = [len(temp[1]) for temp in sort_itme_num]
        # Process out the most interacted items
        # (first sort by the number of users interacting with the item, and finally return the item values in descending order)
        sort_user_num = sorted(self.dict_train_i_u.items(), key=lambda item: len(item[1]), reverse=True)
        self.topK_items = [temp[0] - self.n_users for temp in sort_user_num]  # Guaranteed from 0
        self.topK_items_counts = [len(temp[1]) for temp in sort_user_num]

    def sparse_inter_matrix(self, form):
        return cal_sparse_inter_matrix(self, form)

    def log_info(self, name, interactions, list_u, list_i):
        info = [self.dataset_name]
        inter_num = len(interactions)
        num_u = len(set(list_u))
        num_i = len(set(list_i))
        info.extend(['The number of users: {}'.format(num_u),
                     'Average actions of users: {}'.format(inter_num / num_u)])
        info.extend(['The number of items: {}'.format(num_i),
                     'Average actions of items: {}'.format(inter_num / num_i)])
        info.append('The number of inters: {}'.format(inter_num))
        sparsity = 1 - inter_num / num_u / num_i
        info.append('The sparsity of the dataset: {}%'.format(sparsity * 100))
        self.logger.info('\n====' + name + '====\n' + str('\n'.join(info)))


class Load_dataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)
        self.item_knn_k = config.item_knn_k
        self.user_knn_k = config.user_knn_k
        self.i_mm_image_weight = config.i_mm_image_weight
        self.u_mm_image_weight = config.u_mm_image_weight
        self.all_set = set(range(self.n_users, self.n_users + self.n_items))
        # Print statistical information
        self.log_info("Training", self.train_data, self.train_data[:, 0], self.train_data[:, 1])

        # ***************************************************************************************
        # Prepare four graphs that will be needed later
        # (user co-occurrence graph, user interest graph, item co-occurrence graph, item semantic graph)

        # Construct a user co-occurrence matrix with several items of common interaction between all users
        self.user_co_occ_matrix = load_or_create_matrix(self.logger, "User", " co-occurrence matrix",
                                                        self.dataset_name, "user_co_occ_matrix", creat_co_occur_matrix,
                                                        "user", self.train_data, 0, self.n_users)
        # Construct an item co-occurrence matrix with several users who interact in common between all items
        self.item_co_occ_matrix = load_or_create_matrix(self.logger, "Item", " co-occurrence matrix",
                                                        self.dataset_name, "item_co_occ_matrix", creat_co_occur_matrix,
                                                        "item", self.train_data, self.n_users, self.n_items)

        # Construct a dictionary of user graphs, taking the first 200
        self.dict_user_co_occ_graph = load_or_create_matrix(self.logger, "User", " co-occurrence dict graph",
                                                            self.dataset_name, "dict_user_co_occ_graph",
                                                            creat_dict_graph,
                                                            self.user_co_occ_matrix, self.n_users)
        # Construct a dictionary of item graphs, taking the first 200
        self.dict_item_co_occ_graph = load_or_create_matrix(self.logger, "Item", " co-occurrence dict graph",
                                                            self.dataset_name, "dict_item_co_occ_graph",
                                                            creat_dict_graph,
                                                            self.item_co_occ_matrix, self.n_items)
        # ***************************************************************************************

        # Get the sparse interaction matrix of the training set
        sp_inter_m = sparse_mx_to_torch_sparse_tensor(self.sparse_inter_matrix(form='coo')).to(self.device)
        # Construct a item weight graph
        if self.i_v_feat is not None:  # 4096
            # Construct user visual interest similarity graphs
            self.u_v_interest = tsp.mm(sp_inter_m, self.i_v_feat) / tsp.sum(sp_inter_m, [1]).unsqueeze(dim=1).to_dense()
            u_v_adj = get_knn_adj_mat(self.u_v_interest, self.user_knn_k, self.device)
            i_v_adj = get_knn_adj_mat(self.i_v_feat, self.item_knn_k, self.device)
            self.i_mm_adj = i_v_adj
            self.u_mm_adj = u_v_adj
        if self.i_t_feat is not None:  # 384
            # Construct a user text interest similarity graph
            self.u_t_interest = tsp.mm(sp_inter_m, self.i_t_feat) / tsp.sum(sp_inter_m, [1]).unsqueeze(dim=1).to_dense()
            u_t_adj = get_knn_adj_mat(self.u_t_interest, self.user_knn_k, self.device)
            i_t_adj = get_knn_adj_mat(self.i_t_feat, self.item_knn_k, self.device)
            self.i_mm_adj = i_t_adj
            self.u_mm_adj = u_t_adj
        if self.i_v_feat is not None and self.i_t_feat is not None:
            self.i_mm_adj = self.i_mm_image_weight * i_v_adj + (1.0 - self.i_mm_image_weight) * i_t_adj
            self.u_mm_adj = self.u_mm_image_weight * u_v_adj + (1.0 - self.u_mm_image_weight) * u_t_adj
            del i_t_adj, i_v_adj, u_t_adj, u_v_adj
            torch.cuda.empty_cache()

    # ***************************************************************************************
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        user, pos_item = self.train_data[index]
        neg_item = random.sample(self.all_set - set(self.user_items_dict[user]), 1)[0]
        return torch.LongTensor([user, user]), torch.LongTensor([pos_item, neg_item])


class Load_eval_dataset(BaseDataset):
    def __init__(self, v_or_t, config, eval_dataset):
        super().__init__(config)
        self.eval_dataset = eval_dataset
        self.step = config.eval_batch_size
        self.inter_pr = 0  # Markup of the number of interactions that have been computed
        self.eval_items_per_u = []
        self.eval_len_list = []
        self.train_pos_len_list = []
        self.eval_u = list(set(eval_dataset[:, 0]))  # Total users index
        self.t_data = self.train_data
        self.pos_items_per_u = self.train_items_per_u(self.eval_u)
        self.evalute_items_per_u(self.eval_u)

        self.s_idx = 0  # eval start index  s_idx=pr

        self.eval_users = len(set(eval_dataset[:, 0]))
        self.eval_items = len(set(eval_dataset[:, 1]))

        self.n_inters = eval_dataset.shape[0]  # num_interactions  n_inters=pr_end
        # Print statistical information
        self.log_info(v_or_t, self.eval_dataset, eval_dataset[:, 0], eval_dataset[:, 1])

    def __len__(self):
        return math.ceil(self.n_inters / self.step)

    def __iter__(self):
        return self

    def __next__(self):
        if self.s_idx >= self.n_inters:
            self.s_idx = 0
            self.inter_pr = 0
            raise StopIteration()
        return self._next_batch_data()

    def _next_batch_data(self):
        # Calculate the total number of interactions between the training set from A to B
        inter_cnt = sum(self.train_pos_len_list[self.s_idx: self.s_idx + self.step])
        batch_users = self.eval_u[self.s_idx: self.s_idx + self.step]
        batch_mask_matrix = self.pos_items_per_u[:, self.inter_pr: self.inter_pr + inter_cnt].clone()
        # user_ids to index(Always keep the index value at 0-self.step in preparation for evaluating the mask later on)
        batch_mask_matrix[0] -= self.s_idx
        self.inter_pr += inter_cnt  # Update the starting index of the fetch data interaction data
        self.s_idx += self.step  # Update the starting index of the fetching user before fetching the data interaction data

        return [batch_users, batch_mask_matrix]

    def train_items_per_u(self, eval_users):
        u_ids, i_ids = list(), list()
        for i, u in enumerate(eval_users):
            # Search for the number of items the training set has interacted with in order
            u_ls = self.t_data[np.where(self.t_data[:, 0] == u), 1][0]
            i_len = len(u_ls)
            self.train_pos_len_list.append(i_len)
            u_ids.extend([i] * i_len)
            i_ids.extend(u_ls)
        return torch.tensor([u_ids, i_ids]).type(torch.LongTensor)

    def evalute_items_per_u(self, eval_users):
        for u in eval_users:
            u_ls = self.eval_dataset[np.where(self.eval_dataset[:, 0] == u), 1][0]
            self.eval_len_list.append(len(u_ls))
            self.eval_items_per_u.append(u_ls - self.n_users)  # Items per user interaction
        self.eval_len_list = np.asarray(self.eval_len_list)
