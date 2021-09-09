import os
import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from glob import glob
from collections import defaultdict

from classification_model import Model
from datasets import ProteinSequenceDataset, SimpleSequenceEmbedder
from prefilter.utils.utils import tf_saved_model_collate_fn


def parser():
    ap = ArgumentParser()
    ap.add_argument("--save_prefix", required=True)
    ap.add_argument("--logs_dir", required=True)
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--batch_size", type=int, default=32)
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

    # always going to be false
    hparams['ranking'] = False
    # hparams['learning_rate'] = False
    # hparams['schedule_lr'] = False
    # hparams['step_lr_step_size'] = False
    # hparams['step_lr_decay_factor'] = False
    # del hparams['initial_learning_rate']
    model = Model(**hparams)
    if os.path.splitext(model_path)[1] == '.ckpt':
        checkpoint = torch.load(model_path)
        state_dict = checkpoint['state_dict']
    else:
        state_dict = torch.load(model_path)

    model.load_state_dict(state_dict)
    model.eval()
    return model, hparams


def predict_all_sequences_and_rank(model, test_dataset, decoy_dataset, save_fig):
    # get final classification layer's weights and init a new
    # layer with trained weights

    decoy_scores = []
    total_labels = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    with torch.no_grad():

        for sequence in decoy_dataset:
            sequence = sequence.to(device)
            scores = model.class_act(model(sequence)).squeeze().to('cpu')
            decoy_scores.append(scores.numpy())

        family_id_to_score_and_label = defaultdict(list)
        for sequence, labels in test_dataset:
            labels = labels.squeeze()
            sequence = sequence.to(device)
            embeddings = model(sequence).squeeze()
            scores = model.class_act(embeddings).to('cpu')
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
                num_false_positives_above_threshold_real_sequences = np.count_nonzero(
                    thresholded_real[:, family_class_code])
                threshold_to_true_positives[threshold] += num_true_positives_above_threshold
                threshold_to_false_positives[threshold] += (
                        num_false_positives_above_threshold_decoys + num_false_positives_above_threshold_real_sequences)

    total_sequences = len(test_psd)
    percent_tps_recovered = {k: v / total_labels for k, v in threshold_to_true_positives.items()}
    mean_fps_per_sequence = {k: v / total_sequences for k, v in threshold_to_false_positives.items()}

    fig, ax = plt.subplots(figsize=(13, 10))
    ax.semilogx(list(mean_fps_per_sequence.values()), list(percent_tps_recovered.values()), 'ro--', markersize=10)
    ax.semilogx(list(mean_fps_per_sequence.values()), list(percent_tps_recovered.values()), 'wo', markersize=8)
    ax.set_ylabel('fraction of true positives')
    ax.set_xlabel('mean false positives per sequence')
    ax.set_title('0.5pid clusters, decoys included')
    plt.savefig(save_fig)
    plt.close()


def evaluate_model(model, test_dataset, decoy_dataset, save_fig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sigmoid_threshold_to_tps_passed = defaultdict(int)
    sigmoid_threshold_to_num_passed = defaultdict(int)
    sigmoid_threshold_to_decoys_passed = defaultdict(int)
    sigmoid_threshold_to_tps_missed = defaultdict(int)
    sigmoid_threshold_to_fps_passed = defaultdict(int)

    thresholds = range(10, 101, 5)[::-1]
    thresholds = [t / 100 for t in thresholds]

    model = model.to(device)

    total_true_labels = 0
    total_sequences = 0
    total_decoys = 0

    with torch.no_grad():

        for i, (sequence, labels) in enumerate(test_dataset):
            total_sequences += sequence.shape[0]
            labels = labels.squeeze().numpy()
            total_true_labels += np.count_nonzero(labels)
            sequence = sequence.to(device)
            embeddings = model(sequence).squeeze()
            scores = model.class_act(embeddings).to('cpu').numpy()
            thresholded_scores = scores
            for threshold in thresholds:
                # could refactor, right?
                # i do an unnecessary copy each time. Could just iterate over thresholds
                # from highest to lowest and keep updating thresholded scores
                thresholded_scores[thresholded_scores >= threshold] = 1
                # thresholded_scores[thresholded_scores < threshold] = 0 -> don't modify things below the threshold
                true_positives = np.sum(np.count_nonzero((thresholded_scores == 1).astype(bool) & (labels == 1).astype(bool)))
                false_positives = np.sum(np.count_nonzero((thresholded_scores == 1).astype(bool) & (labels == 0).astype(bool)))
                misses = np.sum(np.count_nonzero((thresholded_scores != 1).astype(bool) & (labels == 1).astype(bool)))
                num_passed = np.sum(np.count_nonzero(thresholded_scores == 1))
                sigmoid_threshold_to_num_passed[threshold] += num_passed
                sigmoid_threshold_to_tps_missed[threshold] += misses
                sigmoid_threshold_to_tps_passed[threshold] += true_positives
                sigmoid_threshold_to_fps_passed[threshold] += false_positives
            print(i, len(test_dataset))

        print('finished real sequences dataset, starting on decoys')
        for sequence, _ in decoy_dataset:
            sequence = sequence.to(device)
            scores = model.class_act(model(sequence)).squeeze().to('cpu').numpy()
            total_decoys += sequence.shape[0]
            thresholded_scores = scores
            for threshold in thresholds:
                thresholded_scores[thresholded_scores >= threshold] = 1
                num_decoys_passed = np.sum(np.count_nonzero(thresholded_scores == 1))
                sigmoid_threshold_to_decoys_passed[threshold] += num_decoys_passed

        fig, ax = plt.subplots(figsize=(13, 10))

        # sigmoid_threshold_to_tps_passed = {k: sum(v) for k, v in sigmoid_threshold_to_tps_passed.items()}
        # sigmoid_threshold_to_decoys_passed = {k: sum(v) for k, v in sigmoid_threshold_to_decoys_passed.items()}
        # sigmoid_threshold_to_num_passed = {k: sum(v) for k, v in sigmoid_threshold_to_num_passed.items()}

        ax.plot(thresholds, [s / total_true_labels for s in sigmoid_threshold_to_tps_passed.values()], 'ro-',
                label='percent tps recovered')
        ax.plot(thresholds, [s / total_sequences for s in sigmoid_threshold_to_fps_passed.values()], 'go-',
                label='mean fps passed per sequence')
        ax.plot(thresholds, [1 - (s / (total_sequences * 253)) for s in sigmoid_threshold_to_num_passed.values()],
                'bo-', label='percent filtered')
        ax1 = ax.twinx()
        ax1.plot(thresholds, [s / total_decoys for s in sigmoid_threshold_to_decoys_passed.values()], 'ko-',
                 label='decoy hits per sequence')

        ax.set_title('accuracy metrics')
        ax.set_ylabel('percent of hits')
        ax1.set_ylabel('hits per sequence')
        ax.set_xlabel('sigmoid threshold')
        ax.legend()
        ax1.legend(loc='lower right')
        plt.savefig(save_fig)
        plt.close()


if __name__ == '__main__':
    args = parser()

    trained_model, hparams = load_model(args.logs_dir, model_path=args.model_path)
    test_files = hparams['test_files']

    pfam_id_to_class_code = hparams['class_code_mapping']
    class_code_to_pfam_id = {v: k for k, v in pfam_id_to_class_code.items()}

    test_psd = ProteinSequenceDataset(test_files,
                                      pfam_id_to_class_code,
                                      evaluating=True,
                                      use_pretrained_model_embeddings=True)

    decoys = SimpleSequenceEmbedder('/home/tc229954/data/prefilter/small-dataset/random_sequences/random_sequences.fa')
    collate_fn = tf_saved_model_collate_fn(args.batch_size)
    test = torch.utils.data.DataLoader(test_psd,
                                       batch_size=args.batch_size,
                                       shuffle=False,
                                       collate_fn=collate_fn)

    decoys = torch.utils.data.DataLoader(decoys,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         collate_fn=collate_fn)

    ranking_figure_name = args.save_prefix + '_rankings.png'
    evaluation_figure_name = args.save_prefix + '_evaluated.png'
    # predict_all_sequences_and_rank(trained_model, test, decoys, ranking_figure_name)
    evaluate_model(trained_model, test, decoys, evaluation_figure_name)
