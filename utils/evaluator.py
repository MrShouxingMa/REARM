import torch
import numpy as np
from utils.metrics import metrics_dict


def evaluate_model(self, epoch, eval_data, t_or_v):
    self.model.eval()
    with torch.no_grad():
        batch_matrix_list = []
        for batch_idx, batched_data in enumerate(eval_data):
            scores = self.model.full_sort_predict(batched_data)
            masked_items = batched_data[1]
            scores[masked_items[0], masked_items[1] - self.model.n_users] = -1e10  # mask out pos items，restore ori_id
            _, top_k_index = torch.topk(scores, max(self.topk), dim=-1)  # nusers x topk
            batch_matrix_list.append(top_k_index)

    pos_items = eval_data.eval_items_per_u
    pos_len_list = eval_data.eval_len_list
    top_k_index = torch.cat(batch_matrix_list, dim=0).cpu().numpy()
    assert len(pos_len_list) == len(top_k_index)
    bool_rec_matrix = []

    for m, n in zip(pos_items, top_k_index):
        bool_rec_matrix.append([True if i in m else False for i in n])
    bool_rec_matrix = np.asarray(bool_rec_matrix)

    # get metrics
    metric_dict = {}
    result_list = cal_metrics(self.metrics, pos_len_list, bool_rec_matrix)
    list_key = []
    for metric, value in zip(self.metrics, result_list):
        for k in self.topk:
            key = '{}@{}'.format(metric, k)
            list_key.append(key) if k == self.topk[-1] else None
            metric_dict[key] = round(value[k - 1], 4)  # Round to 4 decimal points
    valid_score = metric_dict[self.valid_metric] if self.valid_metric else metric_dict['NDCG@20']
    if self.writer is not None:
        for idx in list_key:
            self.writer.add_scalar(t_or_v + "_" + idx, metric_dict[idx], epoch)  # Precision@20,Recall@20,NDCG@20
        self.writer.add_histogram(t_or_v + '_user_visual_distribution', self.model.user_v_prefer, epoch)
        self.writer.add_histogram(t_or_v + '_user_textual_distribution', self.model.user_t_prefer, epoch)
        self.writer.add_embedding(self.model.user_id_embedding.weight, global_step=epoch,
                                  tag=t_or_v + "user_id_embedding")
        self.writer.add_embedding(self.model.item_id_embedding.weight, global_step=epoch,
                                  tag=t_or_v + "item_id_embedding")

    return valid_score, metric_dict


def cal_metrics(topk_metrics, pos_len_list, topk_index):
    result_list = []
    for metric in topk_metrics:
        metric_fuc = metrics_dict[metric]
        result = metric_fuc(topk_index, pos_len_list)
        result_list.append(result)
    return np.stack(result_list, axis=0)
