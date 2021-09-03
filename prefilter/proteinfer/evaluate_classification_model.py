import os
import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from glob import glob
from collections import defaultdict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from classification_model import Model
from datasets import ProteinSequenceDataset, SimpleSequenceEmbedder


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


def performance_by_threshold(test_dataset):
    # i don't care about this granularity.
    # For a set of thresholds, I want to look at the total
    # number of true labels recovered. This is different
    # than what SRE's graph showed since I'm not looking at
    # decoy sequences, and as such have no way to deal with
    # them, i.e. I can't place a sequence into an "unknown" bin
    total_sequences = 0
    total_labels = 0
    threshold_to_recovered_labels = defaultdict(list)
    threshold_to_false_positives = defaultdict(list)
    with torch.no_grad():

        for sequences, labels in test_dataset:

            preds = model.class_act(model(sequences)).squeeze().numpy()
            total_sequences += sequences.shape[0]
            labels = labels.numpy()
            total_true_labels = np.count_nonzero(labels)
            total_labels += total_true_labels

            for threshold in range(10, 100, 5)[::-1]:
                threshold = threshold / 100
                thresholded_preds = preds.copy()
                thresholded_preds[thresholded_preds < threshold] = 0
                thresholded_preds[thresholded_preds >= threshold] = 1

                number_of_labels_recovered = len(np.where((thresholded_preds + labels).ravel() == 2)[0])
                no_trues = thresholded_preds[labels != 1]
                number_of_false_positives = np.count_nonzero(no_trues)

                threshold_to_recovered_labels[threshold].append(number_of_labels_recovered)
                # this needs to be summed then divided by total_labels
                threshold_to_false_positives[threshold].append(number_of_false_positives)  # this needs
                # to be a mean in the end (so I sum the list, then divide by num_sequences)
                # (needs to be mean false positives per search i.e mean false positives per sequence)
                # get total number of true labels recovered

    all_false_positives_per_threshold = {k: sum(v) for k, v in threshold_to_false_positives.items()}
    all_recovered_sequences_per_threshold = {k: sum(v) for k, v in threshold_to_recovered_labels.items()}
    mean_fps_per_sequence = {k: v / total_sequences for k, v in all_false_positives_per_threshold.items()}
    percent_tps_recovered = {k: v / total_labels for k, v in all_recovered_sequences_per_threshold.items()}

    return mean_fps_per_sequence, percent_tps_recovered


def calculate_stats_for_decoy_dataset(decoy_dataset):
    threshold_to_fps = defaultdict(int)

    with torch.no_grad():
        for batch_decoy in decoy_dataset:
            preds = model.class_act(model(batch_decoy)).squeeze().numpy()
            for threshold in range(10, 100, 5)[::-1]:
                threshold = threshold / 100
                thresholded_preds = preds.copy()
                thresholded_preds[thresholded_preds < threshold] = 0
                thresholded_preds[thresholded_preds >= threshold] = 1
                number_of_false_positives = np.count_nonzero(thresholded_preds)
                threshold_to_fps[threshold] += number_of_false_positives

    mean_fp_per_decoy = {k: v / len(decoy_dataset) for k, v in threshold_to_fps.items()}
    return mean_fp_per_decoy


def plothmmer3_plot(mean_fp_per_decoy,
                    mean_fps_per_sequence,
                    percent_tps_recovered):
    fig, ax = plt.subplots(figsize=(13, 10), nrows=2)
    ax[0].semilogx(list(mean_fps_per_sequence.values()), list(percent_tps_recovered.values()), 'ro--', markersize=10)
    ax[0].semilogx(list(mean_fps_per_sequence.values()), list(percent_tps_recovered.values()), 'wo', markersize=8)
    ax[0].set_ylabel('fraction of true positives')
    ax[0].set_xlabel('mean false positives per sequence')
    ax[0].set_title('0.5pid clusters, decoys')
    ax[1].plot(list(mean_fp_per_decoy.keys())[::-1], list(mean_fp_per_decoy.values())[::-1], 'ko--', markersize=10)
    ax[1].plot(list(mean_fp_per_decoy.keys())[::-1], list(mean_fp_per_decoy.values())[::-1], 'wo', markersize=8)
    ax[1].set_xlim([1, 0])
    ax[1].set_xlabel('probability threshold')
    ax[1].set_ylabel('avg. number of decoys let through at threshold')
    plt.savefig('/home/tc229954/testing.png')
    plt.close()


if __name__ == '__main__':
    args = parser()
    test_files = glob(os.path.join(args.test_data, "*test.json"))
    model_dir = args.model_dir
    model, hparams = load_model(model_dir)
    pfam_id_to_class_code = hparams['class_code_mapping']
    class_code_to_pfam_id = {v: k for k, v in pfam_id_to_class_code.items()}
    test_psd = ProteinSequenceDataset(test_files, pfam_id_to_class_code, evaluating=True)
    decoys = SimpleSequenceEmbedder('/home/tc229954/data/prefilter/small-dataset/random_sequences/random_sequences.fa')
    batch_size = 32

    test_dataset = torch.utils.data.DataLoader(test_psd,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               drop_last=False)
    decoy_dataset = torch.utils.data.DataLoader(decoys,
                                                batch_size=batch_size * 2,
                                                shuffle=False,
                                                drop_last=False)

    mean_fps_per_sequence, percent_tps_recovered = performance_by_threshold(test_dataset)
    mean_fp_per_decoy = calculate_stats_for_decoy_dataset(decoy_dataset)

    plothmmer3_plot(mean_fp_per_decoy, mean_fps_per_sequence, percent_tps_recovered)
