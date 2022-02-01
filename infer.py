import argparse
import torch
import torchaudio
from tqdm import tqdm
#import data_loader.data_loaders as module_data
import data_loader.numpy_dataset as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch

import model.diffusion as module_diffusion
import model.network as module_network


from parse_config import ConfigParser

torch.backends.cudnn.benchmark = True

def main(config):
    logger = config.get_logger('infer')

    # setup data_loader instances

    infer_dataset = config.init_obj('infer_dataset', module_data)

    logger.info('Finish initializing datasets')

    sample_rate = config['sample_rate']
    # build model architecture
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    diffusion = config.init_obj('diffusion', module_diffusion, device=device)
    network = config.init_obj('network', module_network)
    model = config.init_obj('arch', module_arch, diffusion, network)
    # prepare model for testing
    model = model.to(device)
    model.eval()
    logger.info(model)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']

    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)



    # get function handles of loss and metrics

    sample_path = config.save_dir/'samples'
    sample_path.mkdir(parents=True, exist_ok=True)

    output_path = sample_path/'output'
    output_path.mkdir(parents=True, exist_ok=True)

    n_samples = len(infer_dataset)
    with torch.no_grad():
        for i in tqdm(range(n_samples)):
            condition = infer_dataset.__getitem__(i)
            condition = condition.to(device)
            # infer from conditional input only
            output = model.infer(condition)

            #
            # save samples, or do something with output here
            #

            name = infer_dataset.getName(i)
            torchaudio.save(output_path/f'{name}.wav', torch.unsqueeze(output, 0).cpu(), sample_rate)

            # computing loss, metrics on test set


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
