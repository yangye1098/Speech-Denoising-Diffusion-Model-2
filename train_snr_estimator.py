import argparse
import collections
import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.snr_estimator as module_arch
import model.segmentor as module_segmentor
from parse_config import ConfigParser
from trainer import SNREstimatorTrainer
from utils import prepare_device


torch.backends.cudnn.benchmark = True

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    tr_dataset = config.init_obj('tr_dataset', module_data, sample_rate=config['sample_rate'], T=config['num_samples'])
    val_dataset = config.init_obj('val_dataset', module_data, sample_rate=config['sample_rate'], T=config['num_samples'])
    tr_data_loader = config.init_obj('data_loader', module_data, tr_dataset)
    val_data_loader = config.init_obj('data_loader', module_data, val_dataset)

    logger.info('Finish initializing datasets')
    #
    device, device_ids = prepare_device(config['n_gpu'])

    logger.info('Finish preparing gpu')
    segmentor = config.init_obj('segmentor', module_segmentor, num_samples=config['num_samples'])
    segmentor = segmentor.to(device)
    model = config.init_obj('arch', module_arch, n_segments=segmentor.n_segments, segment_len=segmentor.F)

    # prepare for (multi-device) GPU training
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    logger.info(model)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    #lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    lr_scheduler = None

    trainer = SNREstimatorTrainer(model, segmentor, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=tr_data_loader,
                      valid_data_loader=val_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Speech denoising diffusion model')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
