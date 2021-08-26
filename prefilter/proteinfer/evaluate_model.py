import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import numpy as np
import yaml
from argparse import ArgumentParser
from glob import glob
from collections import defaultdict


from classification_model import Model
from datasets import ProteinSequenceDataset


def parser():
    ap = ArgumentParser()
    ap.add_argument("--test_data", required=True)
    ap.add_argument("--model_dir", required=True)
    return ap.parse_args()


def load_model(logs_dir):
    yaml_path = os.path.join(logs_dir, 'hparams.yaml')
    models = glob(os.path.join(logs_dir, "*pt"))
    models += glob(os.path.join(logs_dir, "checkpoints", "*ckpt"))
    if len(models) > 1:
        print('{} models found, using the first one'.format(len(models)))

    model_path = models[0]
    with open(yaml_path, 'r') as src:
        hparams = yaml.safe_load(src)

    model = Model(**hparams)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model, hparams


if __name__ == '__main__':

    args = parser()
    test_files = glob(os.path.join(args.test_data, "*test.json"))
    model_dir = args.model_dir
    model, hparams = load_model(model_dir)
    pfam_id_to_class_code = hparams['class_code_mapping']
    class_code_to_pfam_id = {v: k for k, v in pfam_id_to_class_code.items()}
    test_psd = ProteinSequenceDataset(test_files, pfam_id_to_class_code, evaluating=True)
    batch_size = 32
    test_dataset = torch.utils.data.DataLoader(test_psd,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               drop_last=False)
    pfam_id_to_statistics = defaultdict(dict)
    with torch.no_grad():
        for sequences, labels in test_dataset:
            preds = model.class_act(model(sequences)).squeeze().numpy()
            labels = labels.numpy()
            for threshold in range(10, 100, 10)[::-1]:

                threshold = threshold / 100
                thresholded_preds = preds.copy()
                thresholded_preds[thresholded_preds >= threshold] = 1
                thresholded_preds[thresholded_preds < threshold] = 0

                for idx in range(thresholded_preds.shape[0]):
                    total_above_threshold = np.count_nonzero(thresholded_preds[idx])
                    correct_label_idx = np.where(labels[idx] == 1)[0]
                    predicted_label_idx = np.where(thresholded_preds[idx] == 1)[0]
                    pfam_accession_ids = [class_code_to_pfam_id[x] for x in correct_label_idx]
                    for class_code, accession_id in zip(correct_label_idx, pfam_accession_ids):
                        if thresholded_preds[idx][class_code] == 1:
                            if threshold in pfam_id_to_statistics[accession_id]:
                                pfam_id_to_statistics[accession_id][threshold].append((total_above_threshold, True))
                            else:
                                pfam_id_to_statistics[accession_id][threshold] = []
                        else:
                            if threshold in pfam_id_to_statistics[accession_id]:
                                pfam_id_to_statistics[accession_id][threshold].append((total_above_threshold, False))
                            else:
                                pfam_id_to_statistics[accession_id][threshold] = []
            break