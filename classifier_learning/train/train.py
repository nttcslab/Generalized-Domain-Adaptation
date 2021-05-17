import torch
from torch.autograd import Variable
import numpy as np

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, confusion_matrix

def source(feature_extractor, class_classifier, class_criterion, source_dataloader, optimizer, epoch, logger, params):
    # setup models
    feature_extractor.train()
    class_classifier.train()

    # steps
    start_steps = epoch * len(source_dataloader)
    total_steps = params.epochs * len(source_dataloader)

    epoch_log = 'Epoch: {}'.format(epoch)
    logger.info(epoch_log)

    for batch_idx, sdata in enumerate(source_dataloader):
        # prepare the data
        input1, label1, _, _, _ = sdata
        size = input1.shape[0]
        input1, label1 = input1[0:size, :, :, :], label1[0:size]

        if params.use_gpu:
            input1, label1 = Variable(input1.cuda()), Variable(label1.cuda())
        else:
            input1, label1 = Variable(input1), Variable(label1)

        optimizer.zero_grad()

        # compute the output of source domain and target domain
        src_feature = feature_extractor(input1)

        # compute the class loss of src_feature
        class_preds = class_classifier(src_feature)
        class_loss = class_criterion(class_preds, label1)
        class_loss.backward()
        optimizer.step()

        # print loss
        if (batch_idx + 1) % 10 == 0:
            prompts = '[{}/{} ({:.0f}%)]\tClass Loss: {:.6f}'.format(
                batch_idx * len(input1), len(source_dataloader.dataset),
                100. * batch_idx / len(source_dataloader), class_loss.item())
            logger.info(prompts)

def dann_OS(feature_extractor, class_classifier, domain_classifier, class_criterion, domain_criterion,
            source_dataloader, target_dataloader, optimizer, epoch, logger, params, ood_class=-1):
    # setup models
    feature_extractor.train()
    class_classifier.train()
    domain_classifier.train()

    # steps
    start_steps = epoch * len(source_dataloader)
    total_steps = params.epochs * len(source_dataloader)

    epoch_log = 'Epoch: {}'.format(epoch)
    logger.info(epoch_log)

    if epoch == params.change_epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = params.next_lr

    print(confusion_matrix(target_dataloader.dataset.labels, target_dataloader.dataset.pseudo_label))

    if epoch == params.change_epoch or epoch >= params.change_epoch2:
        target_dataloader.dataset.update_pseudo()

    for batch_idx, (sdata, tdata) in enumerate(zip(source_dataloader, target_dataloader)):
        # setup hyperparameters
        p = float(batch_idx + start_steps) / total_steps
        constant = 2. / (1. + np.exp(-params.gamma * p)) - 1

        # prepare the data
        input1, label1, domain1, _    , _      = sdata
        input2, _,      domain2, hist2, index2 = tdata
        size = min((input1.shape[0], input2.shape[0]))
        input1, label1 = input1[0:size, :, :, :], label1[0:size]
        domain1 = domain1[0:size]
        input2 = input2[0:size, :, :, :]
        domain2 = domain2[0:size]
        hist2 = hist2[0:size]
        index2 = index2[0:size]

        if params.use_gpu:
            input1, label1 = Variable(input1.cuda()), Variable(label1.cuda())
            domain1 = Variable(domain1.cuda())
            input2 = Variable(input2.cuda())
            domain2 = Variable(domain2.cuda())
            hist2 = Variable(hist2.cuda()).long()
        else:
            input1, label1 = Variable(input1), Variable(label1)
            domait1 = Variable(domain1)
            input2 = Variable(input2)
            domain2 = Variable(domain2)
            hist2 = Variable(hist2).long()

        # setup optimizer
        optimizer.zero_grad()

        source_labels = domain1
        target_labels = domain2

        if epoch < params.change_epoch:

            # compute the output of source domain and target domain
            src_feature = feature_extractor(input1)
            tgt_feature = feature_extractor(input2)

            # compute the class loss of src_feature
            class_preds = class_classifier(src_feature)
            class_loss = class_criterion(class_preds, label1)

            # compute the domain loss of src_feature and target_feature
            tgt_preds = domain_classifier(tgt_feature, constant)
            src_preds = domain_classifier(src_feature, constant)
            tgt_loss = domain_criterion(tgt_preds, target_labels)
            src_loss = domain_criterion(src_preds, source_labels)
            domain_loss = tgt_loss + src_loss

            loss = class_loss + params.theta * domain_loss

        else:
            input = torch.cat([input1, input2])
            label = torch.cat([label1, hist2])
            domain_label = torch.cat([source_labels, target_labels])

            feat = feature_extractor(input)
            class_preds = class_classifier(feat)
            class_loss = class_criterion(class_preds, label)

            domain_preds = domain_classifier(feat, constant)
            domain_loss = domain_criterion(domain_preds, domain_label)

            class_preds_soft = F.softmax(class_preds)
            avg_probs = torch.mean(class_preds_soft, dim=0)

            p = torch.Tensor(params.prior).cuda()
            prior_loss = -torch.sum(torch.log(avg_probs) * p) * params.prior_weight

            loss = class_loss + params.theta * domain_loss + prior_loss

        loss.backward()
        optimizer.step()

        # print loss
        if (batch_idx + 1) % 10 == 0:
            prompts = '[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tClass Loss: {:.6f}\tDomain Loss: {:.6f}'.format(
                batch_idx * len(input2), len(target_dataloader.dataset),
                100. * batch_idx / len(target_dataloader),
                loss.item(), class_loss.item(),
                domain_loss.item())
            logger.info(prompts)

        with torch.no_grad():
            feature_extractor.eval()
            class_classifier.eval()

            if params.use_gpu:
                input2 = Variable(input2.cuda())
            else:
                input2 = Variable(input2)

            feat2 = feature_extractor(input2)
            output2 = class_classifier(feat2)

            if epoch < params.change_epoch:
                soft2 = F.softmax(output2[:, :-1])
                ent2 = - torch.sum(soft2 * torch.log(soft2 + 1e-6), dim=1)
                pred = output2.max(1)[1]
                ent_sort, _ = torch.sort(ent2, descending=True)
                threshold = ent_sort[int(len(ent_sort) * params.sigma)]
                new_label = torch.where(ent2 > threshold, torch.ones(pred.size()).long().cuda() * ood_class, pred)
            else:
                pred = output2.max(1)[1]
                new_label = pred

            target_dataloader.dataset.update_labels(index2.numpy(), new_label.cpu().numpy())

            feature_extractor.train()
            class_classifier.train()