from simclr import SimCLR
import yaml
from data_aug.dataset_wrapper import DataSetWrapper
from dataset import get_datasets

import numpy as np
import torch
from torchvision.transforms import transforms

from models.baseline_encoder import Encoder
from models.alexnet_simclr import AlexSimCLR
from models.resnet_simclr import ResNetSimCLR

from data_aug.gaussian_blur import GaussianBlur
from easydict import EasyDict

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt

import os
import argparse

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import normalized_mutual_info_score as NMI

parser = argparse.ArgumentParser(description='choose config')
parser.add_argument('--config', type=str, default="./config/Office31/webcam_amazon/grid_3.yaml")
args = parser.parse_args()

def main(args):
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    config = EasyDict(config)

    if config.dataset.parent == 'Digit':
        data_transforms = transforms.Compose([transforms.Resize(32),
                                            transforms.Grayscale(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5],[0.5])])

        model = Encoder()
    elif config.dataset.parent == 'Office31':
        data_transforms = transforms.Compose([transforms.Resize(size=config.model.imsize),
                                              transforms.Grayscale(3),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        if config.model.base_model == 'alexnet':
            model = AlexSimCLR(config.model.out_dim)
        elif config.model.base_model == 'encoder':
            model = Encoder(3, config.model.out_dim)
        else:
            model = ResNetSimCLR(config.model.base_model, config.model.out_dim)

    if isinstance(config.dataset.grid, list):
        config.dataset.grid = min(config.dataset.grid)

    dataset = get_datasets(config.dataset.parent, config.dataset.dset_taples, data_transforms, config.dataset.jigsaw, config.dataset.grid)

    model.eval()
    
    log_dir = os.path.join("record", config.dataset.parent, "_".join(np.array(config.dataset.dset_taples)[:,0]) + "_" + config.dataset.save_name)
    model_path = os.path.join(log_dir, "checkpoints", "model.pth")

    model.load_state_dict(torch.load(model_path))
    model.cuda()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=56, shuffle=False)

    feats = []
    with torch.no_grad():
        for im, _ in dataloader:
            im = im.cuda()
            feat, _ = model(im)

            feats.append(feat.cpu().numpy())
    feats = np.concatenate(feats)

    # tsne plot
    tsne = TSNE(n_components=2, perplexity=30, verbose=1, n_jobs=3)

    feats_decomposit = tsne.fit_transform(feats)

    plt.figure(figsize=(3,3))
    scatter = plt.scatter(feats_decomposit[:, 0], feats_decomposit[:, 1], s=1, c=dataset.domain_labels)
    plt.axis("off")
    plt.savefig(os.path.join(log_dir, "tsne_plot.pdf"), box_inches="tight")

    plt.clf()

    plt.figure(figsize=(3,3))
    scatter = plt.scatter(feats_decomposit[:, 0], feats_decomposit[:, 1], s=1, c=dataset.labels)
    plt.axis("off")
    plt.savefig(os.path.join(log_dir, "tsne_labels_plot.pdf"), box_inches="tight")

    kmeans = KMeans(n_clusters=len(np.unique(dataset.domain_labels)))
    cluster = np.array(kmeans.fit_predict(feats_decomposit))

    np.savetxt(os.path.join(log_dir, "cluster.csv"), cluster, delimiter=",")

    pca = PCA()

    feats_pca = pca.fit_transform(feats)

    plt.clf()

    plt.scatter(feats_pca[:, 0], feats_pca[:, 1], s=1, c=dataset.domain_labels)
    plt.savefig(os.path.join(log_dir, "pca_plot.pdf"))

    plt.clf()

    plt.scatter(feats_pca[:, 0], feats_pca[:, 1], s=1, c=dataset.labels)
    plt.savefig(os.path.join(log_dir, "pca_labels_plot.pdf"))

    gmm = GMM(n_components=len(np.unique(dataset.domain_labels)))

    cluster = np.array(gmm.fit_predict(feats_pca))

    plt.clf()
    plt.scatter(feats_pca[:, 0], feats_pca[:, 1], s = 1, c=cluster)
    plt.savefig(os.path.join(log_dir, "pca_cluster.pdf"))

    np.savetxt(os.path.join(log_dir, "cluster_pca.csv"), cluster, delimiter=",")

    nmi = NMI(dataset.domain_labels, cluster)

    print(f'nmi:{nmi}')

    with open(os.path.join(log_dir, "nmi.txt"), 'w') as f:
        f.write(f'nmi:{nmi}\n')
    
    gmm = GMM(n_components=len(np.unique(dataset.labels)))

    cluster = np.array(gmm.fit_predict(feats_pca))

    plt.clf()
    plt.scatter(feats_pca[:, 0], feats_pca[:, 1], s = 1, c=cluster)
    plt.savefig(os.path.join(log_dir, "pca_cluster_class.pdf"))

    np.savetxt(os.path.join(log_dir, "cluster_pca_class.csv"), cluster, delimiter=",")

    nmi = NMI(dataset.labels, cluster)

    print(f'nmi class:{nmi}')

    with open(os.path.join(log_dir, "nmi_class.txt"), "w") as f:
        f.write(f'nmi:{nmi}\n')

if __name__ == "__main__":
    main(args)
