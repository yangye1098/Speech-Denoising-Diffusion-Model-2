import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import time
from datetime import timedelta
import torchaudio
from model.segmentor import segment_sisnr


class SNREstimatorTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, segmentor, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None,
                 lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.segmentor = segmentor
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            # or debug purpose
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        cfg_trainer = config['trainer']
        self.n_valid_data_batch = cfg_trainer.get('n_valid_data_batch', 2)
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        # get log step
        self.log_step = cfg_trainer.get('log_step', 100)
        self.max_grad_norm = cfg_trainer.get('max_grad_norm', 1.0)

        # only loss for train
        self.train_metrics = MetricTracker('loss', writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        # audio sample dir

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.epoch_start = time.time()
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, (target, condition, _) in enumerate(self.data_loader):
            target, condition = target.to(self.device), condition.to(self.device)
            self.optimizer.zero_grad()
            # use noise in the loss function instead of target (y_0)
            target = self.segmentor(target)
            condition = self.segmentor(condition)
            true_sisnr = segment_sisnr(condition, target)
            output = self.model(condition).squeeze()
            loss = self.criterion(output, true_sisnr)

            loss.backward()
            #grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            if batch_idx>0 and batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.train_metrics.update('loss', loss.item())
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation and (epoch % self.valid_period == 0):
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.logger.debug('')
        self.logger.debug('Valid Epoch: {} started at +{:.0f}s'.format(
            epoch, time.time()-self.epoch_start))
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (target, condition, _) in enumerate(self.valid_data_loader):
                if batch_idx >= self.n_valid_data_batch > 0:
                    break

                target, condition = target.to(self.device), condition.to(self.device)

                target = self.segmentor(target)
                condition = self.segmentor(condition)
                true_sisnr = segment_sisnr(condition, target)
                output = self.model(condition).squeeze()
                loss = self.criterion(output, true_sisnr)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))

                # save the validation output


        self.logger.debug('\nValid Epoch: {} finished at +{:.0f}s'.format(
            epoch, time.time()-self.epoch_start))
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        lapsed = time.time() - self.epoch_start
        base = '[{}/{} | {:.0f}s/{}, ({:.0f}%), ]'
        current = batch_idx
        total = self.len_epoch

        time_left = lapsed * ((total/current) - 1)
        time_left = timedelta(seconds=time_left)
        return base.format(current, total, lapsed, time_left, 100.0 * current / total)
