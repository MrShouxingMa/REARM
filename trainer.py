import torch


def train(self, epoch_idx):
    self.model.train()
    sum_loss = 0.0
    sum_bpr_loss, sum_reg_loss = 0.0, 0.0
    sum_diff_loss, sum_mul_vt_cl_loss = 0.0, 0.0

    step = 0.0

    for batch_idx, interactions in enumerate(self.train_data):
        self.optimizer.zero_grad()
        loss, bpr_loss, reg_loss, mul_vt_cl_loss, diff_loss = self.model.loss(interactions[0], interactions[1])
        if torch.isnan(loss):
            self.logger.info('Loss is nan at epoch: {}, batch index: {}. Exiting.'.format(epoch_idx, batch_idx))
            return loss, torch.tensor(0.0)

        loss.backward()
        self.optimizer.step()

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

    return [mean_loss, mean_bpr_loss, mean_reg_loss, mean_mul_vt_cl_loss, mean_diff_loss]
