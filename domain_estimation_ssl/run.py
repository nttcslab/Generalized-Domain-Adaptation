from simclr import SimCLR
import yaml
from data_aug.dataset_wrapper import DataSetWrapper
from dataset import get_datasets

import numpy as np
import torch
from torchvision.transforms import transforms

from data_aug.gaussian_blur import GaussianBlur
from easydict import EasyDict

import argparse

parser = argparse.ArgumentParser(description='choose config')
parser.add_argument('--config', type=str, default="./config/Office31/webcam_amazon/grid_3.yaml")
args = parser.parse_args()

def main(args):
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    config = EasyDict(config)
    config.config_path = args.config

    if config.dataset.parent == 'Digit':
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(32),
                                        transforms.Grayscale(),
                                        GaussianBlur(kernel_size=int(0.3 * 32), min=0.1, max=2.5),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5],[0.5]),
                                        transforms.Lambda(lambda x: x * (torch.from_numpy(np.random.binomial(1, 0.5, (1)).astype(np.float32) * 2 - 1))),
                                        transforms.Lambda(lambda x: x * (torch.from_numpy(np.random.uniform(low=0.25, high=1.5, size=(1)).astype(np.float32)))),
                                        transforms.Lambda(lambda x: x + (torch.from_numpy(np.random.uniform(low=-0.5, high=0.5, size=(1)).astype(np.float32))))
                                        ])
    elif config.dataset.parent == 'Office31':
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=config.model.imsize, scale=(0.08, 1.0)),
                                              transforms.Grayscale(3),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                              ])

    dataset = get_datasets(config.dataset.parent, config.dataset.dset_taples, data_transforms, config.dataset.jigsaw, config.dataset.grid)

    simclr = SimCLR(dataset, config)
    simclr.train()


if __name__ == "__main__":
    main(args)
