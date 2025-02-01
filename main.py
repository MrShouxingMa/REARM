import os
import torch
import platform
from time import time
from tqdm import tqdm
from trainer import train
import torch.optim as optim
from utils.parser import parse_args
from utils.logger import init_logger
from utils.configurator import Config
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from utils.evaluator import evaluate_model
from utils.data_loader import Load_dataset, Load_eval_dataset
from utils.helper import early_stopping, plot_curve, res_output, stop_log, update_result, sele_para
from model import REARM


class Net:
    def __init__(self, args):
        # Complete initialization of all parameters (including random seeds)
        self.config = Config(args)
        # Use logger
        self.logger = init_logger(self.config)
        self.logger.info(self.config)
        self.logger.info('██Server: \t' + platform.node())
        self.logger.info('██Dir: \t' + os.getcwd() + '\n')
        self.device = self.config.device
        self.model_name = self.config.model_name
        self.dataset_name = self.config.dataset
        self.batch_size = self.config.batch_size
        self.num_workers = self.config.num_workers
        self.learning_rate = self.config.learning_rate
        self.num_epoch = self.config.num_epoch
        self.topk = self.config.topk
        self.metrics = self.config.metrics
        self.valid_metric = self.config.valid_metric
        self.stopping_step = self.config.stopping_step
        self.reg_weight = self.config.reg_weight
        self.cur_step = 0
        self.best_valid_score = -1
        self.best_valid_result = {}
        self.best_test_upon_valid = {}
        # Writer will output to ./runs/ directory by default
        self.writer = SummaryWriter() if self.config.writer else None

        # Perform experimental configurations
        Dataset = Load_dataset(self.config)
        valid_dataset, test_dataset = Dataset.load_eval_data()
        self.train_data = DataLoader(Dataset, batch_size=self.batch_size, shuffle=True,
                                     num_workers=self.num_workers)

        (self.valid_data, self.test_data) = (Load_eval_dataset("Validation", self.config, valid_dataset),
                                             Load_eval_dataset("Testing", self.config, test_dataset))
        self.model = REARM(self.config, Dataset).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), self.learning_rate, weight_decay=self.reg_weight)
        lr_scheduler = self.config.learning_rate_scheduler
        fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        self.lr_scheduler = scheduler
        self.logger.info(self.model)

    def plot_train_loss(self):
        plot_curve(self)

    def run(self):
        run_start_time = time()
        for epoch_idx in tqdm(range(self.num_epoch)):
            train_start_time = time()
            train_loss = train(self, epoch_idx)
            # Save if an exception occurs
            if torch.isnan(train_loss[0]):
                ret_value = {"Recall@20": -1} if self.best_test_upon_valid == {} else self.best_test_upon_valid
                stop_output = '\n ' + str(self.config.dataset) + '  key parameter: ' + sele_para(self.config)
                self.logger.info(stop_output)
                self.logger.info('Loss is nan at epoch: {}; last value is {}Exiting.'.format(epoch_idx, ret_value))
                return ret_value

            self.lr_scheduler.step()

            train_output = res_output(epoch_idx, train_start_time, time(), train_loss, "train")
            self.logger.info(train_output)

            # valid evaluate_model
            valid_start_time = time()
            valid_score, valid_result = evaluate_model(self, epoch_idx, self.valid_data, t_or_v="valid")

            self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                valid_score, self.best_valid_score, self.cur_step, self.stopping_step)

            self.best_valid_result[epoch_idx] = self.best_valid_score
            valid_output = res_output(epoch_idx, valid_start_time, time(), valid_result, t_or_v="valid")
            self.logger.info(valid_output)

            if update_flag:
                # test evaluate_model
                test_start_time = time()
                _, test_result = evaluate_model(self, epoch_idx, self.test_data, t_or_v="test")
                test_score_output = res_output(epoch_idx, test_start_time, time(), test_result, t_or_v="test")
                self.logger.info(test_score_output)
                update_result(self, test_result)

            if stop_flag:
                stop_log(self, epoch_idx, run_start_time)
                break
            else:
                print('patience ==> %d' % (self.stopping_step - self.cur_step))
        return self.best_test_upon_valid


if __name__ == '__main__':
    _args = parse_args()
    model = Net(_args)
    best_score = model.run()
