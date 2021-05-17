"""
Test the model with target domain
"""
import torch
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import accuracy_score

def test(feature_extractor, class_classifier, target_dataloader):
    # setup the network
    feature_extractor.eval()
    class_classifier.eval()

    preds = []
    labels = []
    outputs = []

    for batch_idx, tdata in enumerate(target_dataloader):

        input2, label2, _, _, _ = tdata
        if torch.cuda.is_available():
            input2, label2 = Variable(input2.cuda()), Variable(label2.cuda())
            tgt_labels = Variable(torch.ones((input2.size()[0])).type(torch.LongTensor).cuda())
        else:
            input2, label2 = Variable(input2), Variable(label2)
            tgt_labels = Variable(torch.ones((input2.size()[0])).type(torch.LongTensor))

        feat = feature_extractor(input2)
        output2 = class_classifier(feat)
        pred2 = output2.data.max(1, keepdim=True)[1].squeeze(1)

        outputs.append(output2.cpu().numpy())
        preds.append(pred2.cpu().numpy())
        labels.append(label2.cpu().numpy())

    outputs = np.concatenate(outputs)
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    return outputs, labels, preds