import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import numpy as np
import yaml
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

from argparse import ArgumentParser

from datasets import ProteinSequenceDataset
from classification_model import Model


def parser():
    ap = ArgumentParser()
    ap.add_argument("--logs_dir", required=True)
    ap.add_argument("--gpus", type=int, required=True)
    return ap.parse_args()


def load_model(logs_dir):
    yaml_path = os.path.join(logs_dir, 'hparams.yaml')
    model_path = os.path.join(logs_dir, 'model_50_files.pt')

    with open(yaml_path, 'r') as src:
        hparams = yaml.safe_load(src)

    model = Model(**hparams)
    success = model.load_state_dict(torch.load(model_path))
    return model, hparams


if __name__ == '__main__':

    args = parser()

    model, hparams = load_model(args.logs_dir)
    label_file = hparams['class_code_mapping']
    test_files = hparams['test_files']

    dataset = torch.utils.data.DataLoader(ProteinSequenceDataset(test_files,
                                                                 label_file,
                                                                 evaluating=True), batch_size=32)
    top_n = 1000

    predictions = []
    labels = []

    from sys import stdout

    with torch.no_grad():
        i = 0
        for features, lab in dataset:
            lab = lab.detach().numpy().squeeze()

            preds = torch.sigmoid(model(features)).numpy().squeeze()
            stdout.write('{}\r'.format(i))
            i += 1
            # best_preds = np.argsort(preds)[-top_n:]

            predictions.append(preds.ravel())
            labels.append(lab.ravel())

            # true_label_idx = np.where(labels == 1)[0]
            # num_true_labels = len(true_label_idx)
            # intersection = np.intersect1d(true_label_idx, best_preds)
            # print(len(intersection), num_true_labels, len(intersection)/num_true_labels)
            # exit()

    # metrics.roc_auc_score(np.concatenate(labels), np.concatenate(predictions))
    fpr, tpr, threshold = metrics.roc_curve(np.concatenate(labels),
                                            np.concatenate(predictions))
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('/home/tom/Dropbox/roc_auc_curve.png')
