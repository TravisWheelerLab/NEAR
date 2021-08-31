import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from glob import glob
from collections import defaultdict

from classification_model import Model
from datasets import ProteinSequenceDataset, SimpleSequenceEmbedder


def parser():
    ap = ArgumentParser()
    ap.add_argument("--test_data", required=True)
    ap.add_argument("--save_fig", required=True)
    ap.add_argument("--logs_dir", required=True)
    ap.add_argument("--model_path", default=None)
    return ap.parse_args()


def load_model(logs_dir, model_path):
    yaml_path = os.path.join(logs_dir, 'hparams.yaml')

    if model_path is None:

        models = glob(os.path.join(logs_dir, "*pt"))
        models += glob(os.path.join(logs_dir, "checkpoints", "*ckpt"))
        if len(models) > 1:
            print('{} models found, using the first one'.format(len(models)))

        model_path = models[0]

    with open(yaml_path, 'r') as src:
        hparams = yaml.safe_load(src)

    model = Model(**hparams, ranking=False)
    if os.path.splitext(model_path)[1] == '.ckpt':
        checkpoint = torch.load(model_path)
        state_dict = checkpoint['state_dict']
    else:
        state_dict = torch.load(model_path)

    model.load_state_dict(state_dict)
    model.eval()
    return model, hparams

def predict_all_sequences_and_rank(test_dataset, decoy_dataset, save_fig,
                                   batch_size=32):

    # get final classification layer's weights and init a new
    # layer with trained weights

    decoy_scores = []
    total_labels = 0

    with torch.no_grad():

        for sequence in decoy_dataset:
            scores = model.class_act(model(sequence)).squeeze()
            decoy_scores.append(scores.numpy())

        family_id_to_score_and_label = defaultdict(list)
        for sequence, labels in test_dataset:
            embeddings = model(sequence).squeeze()
            scores = model.class_act(embeddings)
            # The easiest thing to do would be to store scores for each family for each sequence
            # Because each sequences gets rated by each family, just keeping
            # a dictionary of family_id: (score, label) would be easy and be
            # the same as storing all of the scores for each family one-at-a-time
            for labelset, scoreset in zip(labels, scores):
                for unique_label in np.where(labelset == 1)[0]:
                    total_labels += 1
                    family_id_to_score_and_label[unique_label].append(scoreset.numpy())

        # now I have all family ids->scores for each sequence in that family.
        # by concatenating all of the decoys into each score list, I can sort
        # the score list and see where the true labels end up in relation
        # to all of the decoys (and outright misclassifications).

        threshold_to_false_positives = defaultdict(int)
        threshold_to_true_positives = defaultdict(int)

        for family_class_code, list_of_scores in family_id_to_score_and_label.items():
            list_of_scores = np.stack(list_of_scores, axis=0)
            if list_of_scores.ndim < 2:
                list_of_scores = np.expand_dims(list_of_scores, axis=0)
            decoys = np.concatenate(decoy_scores)

            real_sequences_but_not_from_same_family = []
            for cc, scores in family_id_to_score_and_label.items():
                if cc != family_class_code:
                    real_sequences_but_not_from_same_family.extend(scores)

            real_sequences_but_not_from_same_family = np.stack(real_sequences_but_not_from_same_family)

            print(decoys.shape, list_of_scores.shape, real_sequences_but_not_from_same_family.shape)

            for threshold in range(10, 100, 2):

                threshold = threshold / 100
                thresholded_true_scores = list_of_scores.copy()
                thresholded_true_scores[thresholded_true_scores >= threshold] = 1
                thresholded_true_scores[thresholded_true_scores < threshold] = 0

                thresholded_decoys = decoys.copy()
                thresholded_decoys[thresholded_decoys >= threshold] = 1
                thresholded_decoys[thresholded_decoys < threshold] = 0

                thresholded_real = real_sequences_but_not_from_same_family.copy()
                thresholded_real[thresholded_real >= threshold] = 1
                thresholded_real[thresholded_real < threshold] = 0

                num_true_positives_above_threshold = np.count_nonzero(thresholded_true_scores[:, family_class_code])
                num_false_positives_above_threshold_decoys = np.count_nonzero(thresholded_decoys[:, family_class_code])
                num_false_positives_above_threshold_real_sequences = np.count_nonzero(thresholded_real[:, family_class_code])
                threshold_to_true_positives[threshold] += num_true_positives_above_threshold
                threshold_to_false_positives[threshold] += (num_false_positives_above_threshold_decoys)#+ num_false_positives_above_threshold_real_sequences)

    # sequence ideally: shuffled sequence will be thrown out
    # x-axis: cutoff sigmoid probability
    # y-axis: what percent of the things that I hope will get run thru hmmer actually do (percent recovery)
    # y-axis, 2nd plot: what percent of random shit gets puts through
    # set up collection of families

    # now I need to look at #number of families passed through at different sigmoid thresholds
    from pprint import pprint
    total_sequences = len(test_psd)
    percent_tps_recovered = {k: v/total_labels for k, v in threshold_to_true_positives.items()}
    mean_fps_per_sequence = {k: v/total_sequences for k, v in threshold_to_false_positives.items()}
    pprint(percent_tps_recovered)
    pprint(mean_fps_per_sequence)

    fig, ax = plt.subplots(figsize=(13, 10))
    ax.semilogx(list(mean_fps_per_sequence.values()), list(percent_tps_recovered.values()), 'ro--', markersize=10)
    ax.semilogx(list(mean_fps_per_sequence.values()), list(percent_tps_recovered.values()), 'wo', markersize=8)
    ax.set_ylabel('fraction of true positives')
    ax.set_xlabel('mean false positives per sequence')
    ax.set_title('0.5pid clusters, decoys included')
    plt.savefig(save_fig)
    plt.close()

if __name__ == '__main__':
    args = parser()
    test_files = glob(os.path.join(args.test_data, "*test.json"))

    model, hparams = load_model(args.logs_dir, model_path=args.model_path)

    pfam_id_to_class_code = hparams['class_code_mapping']
    class_code_to_pfam_id = {v: k for k, v in pfam_id_to_class_code.items()}

    test_psd = ProteinSequenceDataset(test_files, pfam_id_to_class_code, evaluating=True)
    decoys = SimpleSequenceEmbedder('/home/tc229954/data/prefilter/small-dataset/random_sequences/random_sequences.fa')
    test_dataset = torch.utils.data.DataLoader(test_psd,
                                               batch_size=batch_size,
                                               shuffle=False)

    decoy_dataset = torch.utils.data.DataLoader(decoys,
                                                batch_size=batch_size,
                                                shuffle=False)

    batch_size = 32

    predict_all_sequences_and_rank(test_psd, decoys, args.save_fig)
