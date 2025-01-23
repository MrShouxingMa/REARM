import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn.utils.clip_grad import clip_grad_norm_


# def train(length, epoch, dataloader, model, optimizer, batch_size, writer=None):
def train(self, epoch_idx):
    self.model.train()
    sum_loss = 0.0
    sum_bpr_loss, sum_reg_loss = 0.0, 0.0
    sum_diff_loss, sum_mul_vt_cl_loss = 0.0, 0.0

    step = 0.0
    # bar = tqdm(total=len(self.train_dataset))
    # num_bar = 0  self_vt_cl_loss, mul_vt_cl_loss
    for batch_idx, interactions in enumerate(self.train_data):
        self.optimizer.zero_grad()
        loss, bpr_loss, reg_loss, mul_vt_cl_loss, diff_loss = self.model.loss(interactions[0],
                                                                                                 interactions[1])
        if torch.isnan(loss):
            self.logger.info('Loss is nan at epoch: {}, batch index: {}. Exiting.'.format(epoch_idx, batch_idx))
            return loss, torch.tensor(0.0)

        loss.backward()
        # clip_grad_norm_(self.model.parameters(), max_norm=1, norm_type=2) #梯度裁剪
        self.optimizer.step()  # 更新模型参数

        # bar.update(self.batch_size)  # 进度条
        # num_bar += self.batch_size  # 进度条
        step += 1.0
        sum_loss += loss
        sum_bpr_loss += bpr_loss
        sum_reg_loss += reg_loss
        sum_mul_vt_cl_loss += mul_vt_cl_loss
        sum_diff_loss += diff_loss
    mean_loss = sum_loss / step
    mean_bpr_loss = sum_bpr_loss / step
    mean_reg_loss = sum_reg_loss / step
    mean_mul_vt_cl_loss = sum_mul_vt_cl_loss / step
    mean_diff_loss = sum_diff_loss / step

    if self.writer is not None:
        self.writer.add_scalar('loss/train', mean_loss, epoch_idx)
        self.writer.add_scalar('loss/bpr_loss', mean_bpr_loss, epoch_idx)
        self.writer.add_scalar('loss/reg_loss', mean_reg_loss, epoch_idx)
        self.writer.add_scalar('loss/mul_vt_cl_loss', mean_mul_vt_cl_loss, epoch_idx)
        self.writer.add_scalar('loss/diff_loss', mean_diff_loss, epoch_idx)

    # bar.close()
    return [mean_loss, mean_bpr_loss, mean_reg_loss,  mean_mul_vt_cl_loss, mean_diff_loss]
