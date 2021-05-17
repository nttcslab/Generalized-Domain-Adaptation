import os
import numpy as np
import pandas as pd

from PIL import Image
import torch.utils.data as data
from joblib import Parallel, delayed

import scipy.stats as stats

def load(root, filename, resize=(32, 32)):
    return Image.open(os.path.join(root, filename)).convert("RGB").resize(resize)

class LabeledDataset(data.Dataset):
    def __init__(self, root, filenames, labels, domain_label, num_history=10, resize=(32, 32), transform=None):
        self.root = root
        self.transform = transform
        self.labels = labels
        self.imgs = Parallel(n_jobs=4, verbose=1)([delayed(load)(self.root, filename, resize) for filename in filenames])
        self.domain_labels = [domain_label] * len(self.imgs)
        self.num_history = num_history
        self.pseudo_label = np.zeros(len(self.imgs))
        self.history = np.zeros((len(self.imgs), self.num_history))

    def __getitem__(self, index):
        img, label = self.imgs[index], self.labels[index]
        domain_label = self.domain_labels[index]
        hist = self.pseudo_label[index]
        # hist, _ = stats.mode(self.history[index])

        if self.transform is not None:
            img = self.transform(img)
        
        return img, label, domain_label, hist, index

    def __len__(self):
        return len(self.imgs)
    
    def update_domain_label(self, new_domain_labels):
        self.domain_labels = new_domain_labels

    def update_labels(self, index, results):
        self.history[index] = np.hstack((self.history[index, 1:], results.reshape(-1, 1)))

    def update_pseudo(self):
        target, _ = stats.mode(self.history, axis=1)
        self.pseudo_label = np.squeeze(target)

    def concat_dataset(self, dataset):
        assert self.root == dataset.root
        self.imgs.extend(dataset.imgs)
        self.labels = np.concatenate([self.labels, dataset.labels])
        self.domain_labels = np.concatenate([self.domain_labels, dataset.domain_labels])
        self.history = np.concatenate([self.history, dataset.history])
        self.pseudo_label = np.concatenate([self.pseudo_label, dataset.pseudo_label])

def relabeling_dataset(dataset, union_class): # 複数データセットでクラスラベルを共通化する．(0,1,2), (4,7,9)などで二つ合わせると歯抜けになる時にそれぞれに0,1,2,3,4,5を割り振るための関数．テストセットに対してはout of distributionに-1を振る．
    union_class = sorted(union_class)
    labels = dataset.labels
    unk = len(union_class)
    labels = [union_class.index(l) if l in union_class else unk for l in labels]
    dataset.labels = labels

    return dataset

def get_lut_dataset(parent, dset_taple, domain_label, num_history, train_transform=None, test_transform=None, domain_labels=None):
    # load train
    if parent == 'Digit':
        text_train = os.path.join("data", parent, "{}_train.txt".format(dset_taple[0]))
        text_test = os.path.join("data", parent, "{}_test.txt".format(dset_taple[0]))
        resize=(32,32)
    else:
        text_train = os.path.join("data", parent, "{}.txt".format(dset_taple[0]))
        text_test = os.path.join("data", parent, "{}.txt".format(dset_taple[0]))
        if parent=='Visda':
            resize=(160,160)
        else:
            resize=(256,256)
    df = pd.read_csv(text_train, sep=" ", names=("filename", "label"))
    filenames = df.filename.values
    labels = df.label.values

    labeled_index = [i for i, l in enumerate(labels) if l in dset_taple[1]]
    unlabeled_index = [i for i, l in enumerate(labels) if (l in dset_taple[2]) and (l not in dset_taple[1])]

    labeled_filenames = filenames[labeled_index]
    labeled_labels = labels[labeled_index]
    unlabeled_filenames = filenames[unlabeled_index]
    unlabeled_labels = labels[unlabeled_index]

    root = os.path.join("data", parent, "imgs")
    labeled_dataset = LabeledDataset(root, labeled_filenames, labeled_labels, domain_label, num_history, resize=resize, transform=train_transform)
    unlabeled_dataset = LabeledDataset(root, unlabeled_filenames, unlabeled_labels, domain_label, num_history, resize=resize, transform=train_transform)

    if domain_labels is not None:
        target_domain_labels = domain_labels[:len(filenames)]
        domain_labels = domain_labels[len(filenames):]
        labeled_domain_labels = target_domain_labels[labeled_index]
        unlabeled_domain_labels = target_domain_labels[unlabeled_index]
        labeled_dataset.domain_labels = labeled_domain_labels
        unlabeled_dataset.domain_labels = unlabeled_domain_labels

    # load test
    df = pd.read_csv(text_test, sep=" ", names=("filename", "label"))
    filenames = df.filename.values
    labels = df.label.values

    test_index = [i for i, l in enumerate(labels) if (l in dset_taple[2]) and (l not in dset_taple[1])]

    test_filenames = filenames[test_index]
    test_labels = labels[test_index]

    root = os.path.join("data", parent, "imgs")
    test_dataset = LabeledDataset(root, test_filenames, test_labels, domain_label, resize=resize, transform=test_transform)

    return labeled_dataset, unlabeled_dataset, test_dataset, domain_labels

def get_datasets(parent, dset_taples, num_history, train_transform=None, test_transform=None, args=None):
    if args.clustering_method == "simCLR":
        clustering_filename = "_".join([d[0] for d in dset_taples]) + ".csv"
        domain_labels = np.loadtxt(os.path.join("clustering", parent, "grid_{}".format(args.grid), clustering_filename), delimiter=",").astype(int)
    elif args.clustering_method == "simCLR_OSDA":
        clustering_filename = "_".join([d[0] for d in dset_taples]) + ".csv"
        domain_labels = np.loadtxt(os.path.join("clustering/OSDA", parent, clustering_filename), delimiter=",").astype(int)
    else:
        domain_labels = None

    union_classes = np.unique(sum([t[1] for t in dset_taples],[]))
    td_list = []

    ld, ud, td, domain_labels = get_lut_dataset(parent, dset_taples[0], 0, num_history, train_transform, test_transform, domain_labels)
    td_list.append([dset_taples[0][0], relabeling_dataset(td, union_classes)])
 
    for i, dset_taple in enumerate(dset_taples[1:]):
        ld_t, ud_t, td_t, domain_labels = get_lut_dataset(parent, dset_taple, i + 1, num_history, train_transform, test_transform, domain_labels)

        ld.concat_dataset(ld_t)
        ud.concat_dataset(ud_t)
        td_list.append([dset_taple[0], relabeling_dataset(td_t, union_classes)])

    assert domain_labels is None or len(domain_labels) == 0

    ld = relabeling_dataset(ld, union_classes)

    ld.domain_labels = ld.domain_labels.astype(np.int64)
    ud.domain_labels = ud.domain_labels.astype(np.int64)

    return ld, ud, td_list, len(union_classes)

if __name__ == "__main__":
    from torchvision.transforms import transforms

    parent = "Office31"
    dset_taples = [['amazon', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]],
                  ['dslr'  , [17,18,19,20,21,22,23,24,25,26,27,28,29,30,31], [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]]]

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

    ld, ud, td_list, num_class = get_datasets(parent, dset_taples, train_transform, test_transform)
    img, label, domain, _ = ld[0]
    print(img.size())
    print(label)
    print(domain)
    img, domain, _ = ud[0]
    print(img.size())
    print(domain)
    for _, td in td_list:
        img, label, domain, _ = td[0]
        print(img.size())
        print(label)
        print(domain)
