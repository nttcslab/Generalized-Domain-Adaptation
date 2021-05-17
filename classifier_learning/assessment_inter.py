import glob
import os
import numpy as np
import pandas as pd
import sys

from sklearn.metrics import accuracy_score, confusion_matrix

def assessment(dirname, epoch):
    filename_wild = "*_test_{}.csv".format(epoch)
    path_wild = os.path.join(dirname, filename_wild)

    preds = []
    labels = []
    for path in glob.glob(path_wild):
        df = pd.read_csv(path)
        preds.append(df.preds.values)
        labels.append(df.labels.values)
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    unk_idx = max(labels)
    cmx = confusion_matrix(labels, preds, normalize='true') * 100
    unk = cmx[unk_idx][unk_idx]
    OS = np.mean([cmx[i][i] for i in range(unk_idx + 1) if np.any(labels == i)])
    OS_star = np.mean([cmx[i][i] for i in range(unk_idx) if np.any(labels == i)])
    acc = accuracy_score(labels, preds) * 100

    print(f'unk:{unk}, OS:{OS}, OS_star:{OS_star}, acc:{acc}')

if __name__=='__main__':
    dirname = sys.argv[1]
    best_file = open(os.path.join(dirname, 'best.txt'), 'r')
    best_epoch_txt = best_file.readline()
    best_file.close()
    epoch = int(best_epoch_txt[6:-1])
    # epoch = sys.argv[2]
    assessment(os.path.join(dirname, 'logs'), epoch)
