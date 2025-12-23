import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.helper import get_norm_adj_mat, ssl_loss, topk_sample, cal_diff_loss, propgt_info


class REARM(nn.Module):
    def __init__(self, config, dataset):
        super(REARM, self).__init__()

        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.n_nodes = self.n_users + self.n_items
        self.i_v_feat = dataset.i_v_feat
        self.i_t_feat = dataset.i_t_feat
        self.embedding_dim = config.embedding_dim
        self.feat_embed_dim = config.embedding_dim
        self.dim_feat = self.feat_embed_dim
        self.reg_weight = config.reg_weight
        self.device = config.device
        self.cl_tmp = config.cl_tmp
        self.cl_loss_weight = config.cl_loss_weight
        self.diff_loss_weight = config.diff_loss_weight
        self.n_layers = config.n_layers
        self.num_user_co = config.num_user_co
        self.num_item_co = config.num_item_co
        self.user_aggr_mode = config.user_aggr_mode
        self.n_ii_layers = config.n_ii_layers
        self.n_uu_layers = config.n_uu_layers
        self.k = config.rank
        self.uu_co_weight = config.uu_co_weight
        self.ii_co_weight = config.ii_co_weight

        # Load user and item graphs
        self.topK_users = dataset.topK_users
        self.topK_items = dataset.topK_items
        self.dict_user_co_occ_graph = dataset.dict_user_co_occ_graph
        self.dict_item_co_occ_graph = dataset.dict_item_co_occ_graph
        self.topK_users_counts = dataset.topK_users_counts
        self.topK_items_counts = dataset.topK_items_counts

        self.s_drop = config.s_drop
        self.m_drop = config.m_drop
        self.ly_norm = nn.LayerNorm(self.feat_embed_dim)

        self.self_i_attn1 = nn.MultiheadAttention(1, 1, dropout=self.s_drop, batch_first=True)
        self.self_i_attn2 = nn.MultiheadAttention(1, 1, dropout=self.s_drop, batch_first=True)

        self.mutual_i_attn1 = nn.MultiheadAttention(1, 1, dropout=self.m_drop, batch_first=True)
        self.mutual_i_attn2 = nn.MultiheadAttention(1, 1, dropout=self.m_drop, batch_first=True)

        self.user_id_embedding = nn.Embedding(self.n_users, self.embedding_dim).to(self.device)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim).to(self.device)

        self.prl = nn.PReLU().to(self.device)

        self.cal_bpr = torch.tensor([[1.0], [-1.0]]).to(self.device)

        # load dataset info
        self.norm_adj = get_norm_adj_mat(self, dataset.sparse_inter_matrix(form='coo')).to(self.device)
        # Process to obtain user co-occurrence matrix (n_users*num_user_co)
        self.user_co_graph = topk_sample(self.n_users, self.dict_user_co_occ_graph, self.num_user_co,
                                         self.topK_users, self.topK_users_counts, 'softmax',
                                         self.device)

        # Process to obtain user co-occurrence matrix (n_users*num_user_co)
        self.item_co_graph = topk_sample(self.n_items, self.dict_item_co_occ_graph, self.num_item_co,
                                         self.topK_items, self.topK_items_counts, 'softmax',
                                         self.device)

        # Process to obtain item similarity matrix (n_items* n_items )
        self.i_mm_adj = dataset.i_mm_adj
        # Process to obtain user similarity matrix (n_users* n_users)
        self.u_mm_adj = dataset.u_mm_adj

        # Strengthen ii and uu graphs
        self.stre_ii_graph = self.ii_co_weight * self.item_co_graph + (1.0 - self.ii_co_weight) * self.i_mm_adj
        self.stre_uu_graph = self.uu_co_weight * self.user_co_graph + (1.0 - self.uu_co_weight) * self.u_mm_adj

        if self.i_v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.i_v_feat, freeze=False).to(self.device)
            self.image_i_trs = nn.Linear(self.i_v_feat.shape[1], self.feat_embed_dim)
            self.user_v_prefer = torch.nn.Parameter(dataset.u_v_interest, requires_grad=True).to(self.device)
            self.image_u_trs = nn.Linear(self.i_v_feat.shape[1], self.feat_embed_dim)

        if self.i_t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.i_t_feat, freeze=False).to(self.device)
            self.text_i_trs = nn.Linear(self.i_t_feat.shape[1], self.feat_embed_dim)
            self.user_t_prefer = torch.nn.Parameter(dataset.u_t_interest, requires_grad=True).to(self.device)
            self.text_u_trs = nn.Linear(self.i_t_feat.shape[1], self.feat_embed_dim)

        # MLP(input_dim, feature_dim, hidden_dim, output_dim)
        self.mlp_u1 = MLP(self.feat_embed_dim, self.feat_embed_dim * self.k, self.feat_embed_dim * self.k, self.device)
        self.mlp_u2 = MLP(self.feat_embed_dim, self.feat_embed_dim * self.k, self.feat_embed_dim * self.k, self.device)
        self.mlp_i1 = MLP(self.feat_embed_dim, self.feat_embed_dim * self.k, self.feat_embed_dim * self.k, self.device)
        self.mlp_i2 = MLP(self.feat_embed_dim, self.feat_embed_dim * self.k, self.feat_embed_dim * self.k, self.device)
        self.meta_netu = nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim, bias=True)  # Knowledge compression
        self.meta_neti = nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim, bias=True)  # Knowledge compression

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.user_id_embedding.weight, std=0.1)
        nn.init.normal_(self.item_id_embedding.weight, std=0.1)

        nn.init.xavier_normal_(self.image_i_trs.weight)
        nn.init.xavier_normal_(self.text_i_trs.weight)
        nn.init.xavier_normal_(self.image_u_trs.weight)
        nn.init.xavier_normal_(self.text_u_trs.weight)

    def forward(self):
        # Uniform feature dimensions for multi-modal feature information on item
        trs_item_v_feat = self.image_i_trs(self.image_embedding.weight)
        trs_item_t_feat = self.text_i_trs(self.text_embedding.weight)  # num_items * 64

        trs_user_v_prefer = self.image_u_trs(self.user_v_prefer)
        trs_user_t_prefer = self.text_u_trs(self.user_t_prefer)  # num_items * 64

        # ====================================================================================
        # Homography Relation Learning
        # ====================================================================================
        # Item homogeneous relational learning
        item_v_t = torch.cat((trs_item_v_feat, trs_item_t_feat), dim=-1)
        item_id_v_t = torch.cat((self.item_id_embedding.weight, item_v_t), dim=-1)
        item_id_v_t = propgt_info(item_id_v_t, self.n_ii_layers, self.stre_ii_graph, last_layer=True)
        item_id_v_t = F.normalize(item_id_v_t)

        item_id_ii = item_id_v_t[:, :self.embedding_dim]
        gnn_i_v_feat = item_id_v_t[:, self.feat_embed_dim:-self.feat_embed_dim]
        gnn_i_t_feat = item_id_v_t[:, -self.feat_embed_dim:]

        # User homogeneous relational learning
        user_v_t = torch.cat((trs_user_v_prefer, trs_user_t_prefer), dim=-1)
        user_id_v_t = torch.cat((self.user_id_embedding.weight, user_v_t), dim=-1)
        user_id_v_t = propgt_info(user_id_v_t, self.n_uu_layers, self.stre_uu_graph, last_layer=True)

        user_id_v_t = F.normalize(user_id_v_t)
        user_id_uu = user_id_v_t[:, :self.embedding_dim]
        gnn_u_v_prefer = user_id_v_t[:, self.embedding_dim:-self.feat_embed_dim]
        gnn_u_t_prefer = user_id_v_t[:, -self.feat_embed_dim:]

        # ====================================================================================
        # Item Feature Attention Integration
        # ====================================================================================
        # Item visual features self-attention
        item_v_feat, _ = self.self_i_attn1(gnn_i_v_feat.unsqueeze(2), gnn_i_v_feat.unsqueeze(2),
                                           gnn_i_v_feat.unsqueeze(2), need_weights=False)
        item_v_feat = self.ly_norm(gnn_i_v_feat + item_v_feat.squeeze())
        item_v_feat = self.prl(item_v_feat)

        # Item text features self-attention
        item_t_feat, _ = self.self_i_attn2(gnn_i_t_feat.unsqueeze(2), gnn_i_t_feat.unsqueeze(2),
                                           gnn_i_t_feat.unsqueeze(2), need_weights=False)
        item_t_feat = self.ly_norm(gnn_i_t_feat + item_t_feat.squeeze())
        item_t_feat = self.prl(item_t_feat)

        # ---------------------------------------------------------------------------------------
        # Item text to visual cross-attention
        i_t2v_feat, _ = self.mutual_i_attn1(item_t_feat.unsqueeze(2), item_v_feat.unsqueeze(2),
                                            item_v_feat.unsqueeze(2), need_weights=False)
        item_t2v_feat = self.ly_norm(item_v_feat + i_t2v_feat.squeeze())
        item_t2v_feat = self.prl(item_t2v_feat)

        # Item visual to text cross-attention
        i_v2t_feat, _ = self.mutual_i_attn2(item_v_feat.unsqueeze(2), item_t_feat.unsqueeze(2),
                                            item_t_feat.unsqueeze(2), need_weights=False)
        item_v2t_feat = self.ly_norm(item_t_feat.squeeze() + i_v2t_feat.squeeze())
        item_v2t_feat = self.prl(item_v2t_feat)

        user_v_prefer = self.prl(gnn_u_v_prefer)  # (num_items* 64)
        user_t_prefer = self.prl(gnn_u_t_prefer)

        # ====================================================================================
        # Heterography Relation Learning
        # ====================================================================================
        # Item feature splicing with total attentions
        item_v_t_feat = torch.cat((item_t2v_feat, item_v2t_feat), dim=-1)  # (num_items* 128)
        user_v_t_prefer = torch.cat((user_v_prefer, user_t_prefer), dim=-1)  # (num_user* 128)
        ego_feat_prefer = torch.cat((user_v_t_prefer, item_v_t_feat), dim=0)  # (num_users+num_items）* 128)
        self.fin_feat_prefer = propgt_info(ego_feat_prefer, self.n_layers, self.norm_adj)

        ego_id_embed = torch.cat((user_id_uu, item_id_ii), dim=0)  # (num_users+num_items）* 64)
        fin_id_embed = propgt_info(ego_id_embed, self.n_layers, self.norm_adj)

        share_knowldge = self.meta_extra_share(fin_id_embed, self.fin_feat_prefer)  # (num_users+num_items）* 64)

        fin_v = self.prl(self.fin_feat_prefer[:, :self.embedding_dim]) + fin_id_embed
        fin_t = self.prl(self.fin_feat_prefer[:, self.embedding_dim:]) + fin_id_embed
        fin_share = self.prl(share_knowldge) + fin_id_embed

        temp_full_feat_prefer = torch.cat((fin_v, fin_t), dim=-1)
        representation = torch.cat((temp_full_feat_prefer, fin_share), dim=-1)

        return representation

    def loss(self, user_tensor, item_tensor):
        user_tensor_flatten = user_tensor.view(-1)
        item_tensor_flatten = item_tensor.view(-1)
        out = self.forward()
        user_rep = out[user_tensor_flatten]
        item_rep = out[item_tensor_flatten]

        score = torch.sum(user_rep * item_rep, dim=1).view(-1, 2)
        bpr_score = torch.matmul(score, self.cal_bpr)
        bpr_loss = -torch.mean(nn.LogSigmoid()(bpr_score))

        # Loss of multi-modal feature contrasts
        i_mul_vt_cl_loss = ssl_loss(self.fin_feat_prefer[:, :self.feat_embed_dim],
                                    self.fin_feat_prefer[:, -self.feat_embed_dim:], item_tensor_flatten, self.cl_tmp)
        u_mul_vt_cl_loss = ssl_loss(self.fin_feat_prefer[:, :self.feat_embed_dim],
                                    self.fin_feat_prefer[:, -self.feat_embed_dim:], user_tensor_flatten, self.cl_tmp)
        mul_vt_cl_loss = self.cl_loss_weight * (i_mul_vt_cl_loss + u_mul_vt_cl_loss)

        # Modal-unique orthogonal constraint
        mul_u_diff_loss = cal_diff_loss(self.fin_feat_prefer, user_tensor, self.feat_embed_dim)
        mul_i_diff_loss = cal_diff_loss(self.fin_feat_prefer, item_tensor, self.feat_embed_dim)
        mul_diff_loss = self.diff_loss_weight * (mul_i_diff_loss + mul_u_diff_loss)

        reg_loss = 0  # Realized in AdamW
        total_loss = bpr_loss + reg_loss + mul_vt_cl_loss + mul_diff_loss

        return total_loss, bpr_loss, reg_loss, mul_vt_cl_loss, mul_diff_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        representation = self.forward()
        u_reps, i_reps = torch.split(representation, [self.n_users, self.n_items], dim=0)
        score_mat_ui = torch.matmul(u_reps[user], i_reps.t())
        return score_mat_ui

    def meta_extra_share(self, id_embed, prefer_or_feat):
        u_id_embed = id_embed[:self.n_users, :]
        i_id_embed = id_embed[self.n_users:, :]

        u_v_t = prefer_or_feat[:self.n_users, :]
        i_v_t = prefer_or_feat[self.n_users:, :]

        # meta-knowlege extraction
        u_knowldge = self.meta_netu(u_v_t).detach()
        i_knowldge = self.meta_neti(i_v_t).detach()

        """ Personalized transformation parameter matrix """
        # Low rank matrix decomposition
        metau1 = self.mlp_u1(u_knowldge).reshape(-1, self.feat_embed_dim, self.k)  # N_u*d*k    [19445, 64, 3]
        metau2 = self.mlp_u2(u_knowldge).reshape(-1, self.k, self.feat_embed_dim)  # N_u*k*d   [19445, 3, 64]
        metai1 = self.mlp_i1(i_knowldge).reshape(-1, self.feat_embed_dim, self.k)  # N_i*d*k   [7050, 64, 3]
        metai2 = self.mlp_i2(i_knowldge).reshape(-1, self.k, self.feat_embed_dim)  # N_i*k*d   [7050, 3,64]
        meta_biasu = torch.mean(metau1, dim=0)  # d*k [64, 3]
        meta_biasu1 = torch.mean(metau2, dim=0)  # k*d    [3,64]
        meta_biasi = torch.mean(metai1, dim=0)  # [64, 3]
        meta_biasi1 = torch.mean(metai2, dim=0)  # [3, 64]
        low_weightu1 = F.softmax(metau1 + meta_biasu, dim=1)
        low_weightu2 = F.softmax(metau2 + meta_biasu1, dim=1)
        low_weighti1 = F.softmax(metai1 + meta_biasi, dim=1)
        low_weighti2 = F.softmax(metai2 + meta_biasi1, dim=1)

        # The learned matrix as the weights of the transformed network Equal to a two-layer linear network;
        u_middle_knowldge = torch.sum(torch.multiply(u_id_embed.unsqueeze(-1), low_weightu1), dim=1)
        u_share_knowldge = torch.sum(torch.multiply(u_middle_knowldge.unsqueeze(-1), low_weightu2), dim=1)
        i_middle_knowldge = torch.sum(torch.multiply(i_id_embed.unsqueeze(-1), low_weighti1), dim=1)
        i_share_knowldge = torch.sum(torch.multiply(i_middle_knowldge.unsqueeze(-1), low_weighti2), dim=1)

        share_knowldge = torch.cat((u_share_knowldge, i_share_knowldge), dim=0)
        return share_knowldge


class MLP(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, output_dim, device):
        super(MLP, self).__init__()
        self.device = device
        self.linear_pre = nn.Linear(input_dim, feature_dim, bias=True)
        self.prl = nn.PReLU().to(self.device)
        self.linear_out = nn.Linear(feature_dim, output_dim, bias=True)

    def forward(self, data):
        x = self.prl(self.linear_pre(data))
        x = self.linear_out(x)
        x = F.normalize(x, p=2, dim=-1)
        return x

