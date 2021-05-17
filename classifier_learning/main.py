"""
Main script for models
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

import numpy as np
import pandas as pd

from models import models, resnet, alexnet
from train import test, train
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, accuracy_score

import argparse, sys, os

import torch
from torch.autograd import Variable

import time
import yaml
import easydict
import shutil

from dataset import get_datasets

import logging
from logging import getLogger,Formatter,StreamHandler,FileHandler,DEBUG
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='choose config')
parser.add_argument('--config', type=str, default="./config/sample.yaml")
parser.add_argument('--outname', type=str, default="output")
args = parser.parse_args()

config_file = args.config
outname = args.outname

args = yaml.load(open(config_file))
args = easydict.EasyDict(args)

outdir = os.path.join("record", outname)
args.checkpoint_dir = os.path.join(outdir, "checkpoints")
args.log_dir = os.path.join(outdir, "logs")

os.makedirs(outdir, exist_ok=True)
os.makedirs(args.checkpoint_dir, exist_ok=True)
os.makedirs(args.log_dir, exist_ok=True)

shutil.copy(config_file, os.path.join(outdir, "config.yaml"))

record_log = os.path.join(args.log_dir, "prompts.log")
record_test = os.path.join(args.log_dir, "{}_test_{}.csv")
record_output = os.path.join(args.log_dir, "{}_outpout_{}.csv")

if os.path.exists(record_log):
    os.remove(record_log)

# logger
logger = getLogger("show_loss_accuarcy")
formatter = Formatter('%(asctime)s [%(levelname)s] \n%(message)s')
handlerSh = StreamHandler()
handlerFile = FileHandler(record_log)
handlerSh.setFormatter(formatter)
handlerSh.setLevel(DEBUG)
handlerFile.setFormatter(formatter)
handlerFile.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handlerSh)
logger.addHandler(handlerFile)

# tensorboard setup
writer = SummaryWriter(outdir)

def get_model(parent, model, num_class, num_domains, pretrained=True):
    if parent == 'Digit':
        feature_extractor = models.SVHN_Extractor()
        class_classifier = models.SVHN_Class_classifier(num_classes=num_class)
        domain_classifier = models.SVHN_Domain_classifier(num_domains=num_domains)
    else:
        if model == 'resnet':
            feature_extractor, class_classifier, domain_classifier = resnet.get_models(num_class, num_domains, pretrained)
        elif model == 'alexnet':
            feature_extractor, class_classifier, domain_classifier = alexnet.get_models(num_class, num_domains, pretrained)
        else:
            raise ValueError('args.model should be resnet or alexnet')

    return feature_extractor, class_classifier, domain_classifier

def entropy(output):
    soft = softmax(output, axis=1)
    return - np.sum(soft * np.log(soft + 1e-6), axis=1)

def setup_datasets(args):
    parent = args.parent
    dset_taples = args.dset_taples

    if parent == 'Digit':
        train_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    ld, ud, td_list, num_class = get_datasets(parent, dset_taples, args.num_history, train_transform, test_transform, args)
    return ld, ud, td_list, num_class

def eval(feature_extractor, class_classifier, td_list, num_class, epoch, logger):
    labels_list = []
    preds_list = []

    accuracy_list = []
    for domain, td in td_list:
        if len(td) == 0:
            continue
        tgt_test_dataloader = torch.utils.data.DataLoader(td, batch_size=56, shuffle=False, num_workers=1)
        with torch.no_grad():
            outputs, labels, preds = test.test(feature_extractor, class_classifier, tgt_test_dataloader)

        accuracy = accuracy_score(labels, preds)
        writer.add_scalar('data/acc/{}'.format(domain), accuracy, epoch)
        accuracy_list.append((domain, accuracy))

        prompt = 'Target Accuracy [{}]: {}/{} ({:.4f}%)\n'.format(domain, np.sum(np.equal(preds, labels)), len(labels), accuracy * 100.)
        logger.info(prompt)

        ents = entropy(outputs[:, :-1])

        labels_list.append(labels)
        preds_list.append(preds)

        # save log
        df = pd.DataFrame({'preds': preds, 'labels': labels, 'entropys': ents})
        df.to_csv(record_test.format(domain, epoch), sep=',')

        # save output result
        if args.save_output:
            np.savetxt(record_output.format(domain, epoch), outputs, delimiter=",")

    labels_list = np.concatenate(labels_list)
    preds_list = np.concatenate(preds_list)

    total_acc = accuracy_score(labels_list, preds_list)
    logger.info("total_accuracy:{}".format(total_acc))
    writer.add_scalar('data/acc/total', total_acc, epoch)

    mtx = confusion_matrix(labels_list, preds_list, labels=np.arange(num_class))
    logger.info("confusion_matrix\n" + str(mtx))

    return total_acc, accuracy_list, mtx

def run_source(ld, ud, td_list, num_class):
    src_train_dataloader = torch.utils.data.DataLoader(ld, args.batch_size, shuffle=True, num_workers=2)

    class_criterion = nn.CrossEntropyLoss()

    feature_extractor, class_classifier, _, = get_model(args.parent, args.model, num_class, args.num_domains, args.pretrained)

    if args.use_gpu:
        feature_extractor.cuda()
        class_classifier.cuda()

    if args.optim == "Adam":
        optimizer = optim.Adam([{'params': feature_extractor.parameters(), 'lr': args.lr},
                            {'params': class_classifier.parameters(), 'lr': args.lr}], lr=args.lr)
    elif args.optim == "momentum":
        optimizer = optim.SGD([{'params': feature_extractor.parameters(), 'lr': args.lr},
                            {'params': class_classifier.parameters(), 'lr': args.lr}], lr=args.lr, momentum=0.9, weight_decay=0.0005)
    else:
        raise NotImplementedError

    best = 0.0

    for epoch in range(args.epochs):
        train.source(feature_extractor, class_classifier, class_criterion, src_train_dataloader, optimizer, epoch, logger, args)

        total_acc, accuracy_list, mtx = eval(feature_extractor, class_classifier, td_list, num_class, epoch, logger)

        # save model
        torch.save(feature_extractor.state_dict(), os.path.join(args.checkpoint_dir, "feature_extractor_latest.tar"))
        torch.save(class_classifier.state_dict(), os.path.join(args.checkpoint_dir, "class_classifier_latest.tar"))

        if total_acc > best:
            best = total_acc
            torch.save(feature_extractor.state_dict(), os.path.join(args.checkpoint_dir, "feature_extractor_best.tar"))
            torch.save(class_classifier.state_dict(), os.path.join(args.checkpoint_dir, "class_classifier_best.tar"))

            with open(os.path.join(outdir, "best.txt"), "w") as f:
                f.write("Epoch {}\n".format(epoch))
                for domain, acc in accuracy_list:
                    f.write("{}:{}\n".format(domain, acc * 100.0))
                f.write("total acc:{}\n".format(total_acc))
                f.write("confusion_matrix\n" + str(mtx))

def run_dann_OS(ld, ud, td_list, num_class):
    test_transform = td_list[0][1].transform

    true_domain_label = np.concatenate([ld.domain_labels, ud.domain_labels])
    true_class_label = np.concatenate([ld.labels, ud.labels])

    # ld.domain_labels = np.zeros(len(ld.domain_labels), dtype=np.int64)
    # ud.domain_labels = np.zeros(len(ud.domain_labels), dtype=np.int64)

    if args.clustering_method == "simCLR" and args.parent is not 'Digit':
        logger.info('use weighted sampler')
        from collections import Counter
        domain_labels = np.concatenate([ld.domain_labels, ud.domain_labels])
        domain_count = list(Counter(domain_labels).values())
        weight = np.sum(domain_count).astype(float) / np.array(domain_count).astype(float)
        weight_src = torch.DoubleTensor(weight[ld.domain_labels])
        weight_tgt = torch.DoubleTensor(weight[ud.domain_labels])
        sampler_src = torch.utils.data.WeightedRandomSampler(weight_src, len(weight_src))
        sampler_tgt = torch.utils.data.WeightedRandomSampler(weight_tgt, len(weight_tgt))
        src_train_dataloader = torch.utils.data.DataLoader(ld, args.batch_size, sampler=sampler_src, num_workers=2)
        tgt_train_dataloader = torch.utils.data.DataLoader(ud, args.batch_size, sampler=sampler_tgt, num_workers=2)
    else:
        src_train_dataloader = torch.utils.data.DataLoader(ld, args.batch_size, shuffle=True, num_workers=2)
        tgt_train_dataloader = torch.utils.data.DataLoader(ud, args.batch_size, shuffle=True, num_workers=2)

    # init criterions
    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()

    feature_extractor, class_classifier, domain_classifier = get_model(args.parent, args.model, num_class, args.num_domains)

    if args.use_gpu:
        feature_extractor.cuda()
        class_classifier.cuda()
        domain_classifier.cuda()

    if args.optim == "Adam":
        optimizer = optim.Adam([{'params': feature_extractor.parameters(), 'lr': args.lr},
                            {'params': class_classifier.parameters(), 'lr': args.lr * args.lr_weight},
                            {'params': domain_classifier.parameters(), 'lr': args.lr * args.lr_weight}], lr=args.lr)
    elif args.optim == "momentum":
        optimizer = optim.SGD([{'params': feature_extractor.parameters(), 'lr': args.lr},
                            {'params': class_classifier.parameters(), 'lr': args.lr * args.lr_weight},
                            {'params': domain_classifier.parameters(), 'lr': args.lr * args.lr_weight}], lr=args.lr, momentum=0.9, weight_decay=0.0005)
    else:
        raise NotImplementedError

    best = 0.0

    for epoch in range(args.epochs):
        train.dann_OS(feature_extractor, class_classifier, domain_classifier, class_criterion, domain_criterion,src_train_dataloader, tgt_train_dataloader, optimizer, epoch, logger, args, ood_class=num_class-1)

        total_acc, accuracy_list, mtx = eval(feature_extractor, class_classifier, td_list, num_class, epoch, logger)

        # save model
        torch.save(feature_extractor.state_dict(), os.path.join(args.checkpoint_dir, "feature_extractor_latest.tar"))
        torch.save(class_classifier.state_dict(), os.path.join(args.checkpoint_dir, "class_classifier_latest.tar"))
        torch.save(domain_classifier.state_dict(), os.path.join(args.checkpoint_dir, "domain_classifier_latest.tar"))

        if total_acc > best:
            best = total_acc
            torch.save(feature_extractor.state_dict(), os.path.join(args.checkpoint_dir, "feature_extractor_best.tar"))
            torch.save(class_classifier.state_dict(), os.path.join(args.checkpoint_dir, "class_classifier_best.tar"))
            torch.save(domain_classifier.state_dict(), os.path.join(args.checkpoint_dir, "domain_classifier_best.tar"))

            with open(os.path.join(outdir, "best.txt"), "w") as f:
                f.write("Epoch {}\n".format(epoch))
                for domain, acc in accuracy_list:
                    f.write("{}:{}\n".format(domain, acc * 100.0))
                f.write("total acc:{}\n".format(total_acc))
                f.write("confusion_matrix\n" + str(mtx))

def main(args):
    ld, ud, td_list, num_class = setup_datasets(args)
    if args.training_mode == "source":
        run_source(ld, ud, td_list, num_class + 1)
    elif args.training_mode == "dann_OS":
        run_dann_OS(ld, ud, td_list, num_class + 1)

if __name__ == '__main__':
    main(args)
