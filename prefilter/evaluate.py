# pylint: disable=no-member
import os

import torch
import numpy as np
import yaml
import torchmetrics
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from glob import glob
from collections import defaultdict

from prefilter.models import Prot2Vec

from prefilter.utils import ProteinSequenceDataset, SimpleSequenceIterator, pad_batch, stack_batch

def precision_and_recall(cmat):
    res = cmat.compute().detach().cpu().numpy()
    rec = res / np.sum(res, axis=0, keepdims=True)
    prec = res / np.sum(res, axis=1, keepdims=True)
    r = np.diag(rec)
    p = np.diag(prec)
    return np.nanmean((2*r*p)/(r+p)), r, p


def load_model(logs_dir, model_path=None):
    yaml_path = os.path.join(logs_dir, "hparams.yaml")

    if model_path is None:

        models = glob(os.path.join(logs_dir, "*pt"))
        models += glob(os.path.join(logs_dir, "checkpoints", "*ckpt"))
        if len(models) > 1:
            print("{} models found, using the first one".format(len(models)))

        model_path = models[0]

    with open(yaml_path, "r") as src:
        hparams = yaml.safe_load(src)

    # always going to be false
    if "res_block_n_filters" in hparams:
        model = Prot2Vec(**hparams)
    if os.path.splitext(model_path)[1] == ".ckpt":
        checkpoint = torch.load(model_path)
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = torch.load(model_path)

    success = model.load_state_dict(state_dict)
    model.eval()
    return model, hparams


def predict_all_sequences_and_rank(model, test_dataset, decoy_dataset, save_fig):
    # get final classification layer's weights and init a new
    # layer with trained weights

    decoy_scores = []
    total_labels = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():

        for sequence in decoy_dataset:
            sequence = sequence.to(device)
            scores = model.class_act(model(sequence)).squeeze().to("cpu")
            decoy_scores.append(scores.numpy())

        family_id_to_score_and_label = defaultdict(list)
        for sequence, labels in test_dataset:
            labels = labels.squeeze()
            sequence = sequence.to(device)
            embeddings = model(sequence).squeeze()
            scores = model.class_act(embeddings).to("cpu")
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

            real_sequences_but_not_from_same_family = np.stack(
                real_sequences_but_not_from_same_family
            )

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

                num_true_positives_above_threshold = np.count_nonzero(
                    thresholded_true_scores[:, family_class_code]
                )
                num_false_positives_above_threshold_decoys = np.count_nonzero(
                    thresholded_decoys[:, family_class_code]
                )
                num_false_positives_above_threshold_real_sequences = np.count_nonzero(
                    thresholded_real[:, family_class_code]
                )
                threshold_to_true_positives[
                    threshold
                ] += num_true_positives_above_threshold
                threshold_to_false_positives[threshold] += (
                    num_false_positives_above_threshold_decoys
                    + num_false_positives_above_threshold_real_sequences
                )

    total_sequences = len(test_dataset)
    percent_tps_recovered = {
        k: v / total_labels for k, v in threshold_to_true_positives.items()
    }
    mean_fps_per_sequence = {
        k: v / total_sequences for k, v in threshold_to_false_positives.items()
    }

    fig, ax = plt.subplots(figsize=(13, 10))
    ax.semilogx(
        list(mean_fps_per_sequence.values()),
        list(percent_tps_recovered.values()),
        "ro--",
        markersize=10,
    )
    ax.semilogx(
        list(mean_fps_per_sequence.values()),
        list(percent_tps_recovered.values()),
        "wo",
        markersize=8,
    )
    ax.set_ylabel("fraction of true positives")
    ax.set_xlabel("mean false positives per sequence")
    ax.set_title("0.5pid clusters, decoys included")
    plt.savefig(save_fig)
    plt.close()


def evaluate_model(model, test_dataset, decoy_dataset, save_fig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        for i, batch in enumerate(test_dataset):
            if len(batch) == 3:
                sequence, sequence_masks, labels = batch
            else:
                sequence, labels = batch
                sequence_masks = None

            total_sequences += sequence.shape[0]
            labels = labels.squeeze().numpy()
            total_true_labels += np.count_nonzero(labels)
            sequence = sequence.to(device)
            embeddings = model(sequence, sequence_masks).squeeze()
            scores = model.class_act(embeddings).to("cpu").numpy()
            thresholded_scores = scores
            for threshold in thresholds:
                # could refactor, right?
                # i do an unnecessary copy each time. Could just iterate over thresholds
                # from highest to lowest and keep updating thresholded scores
                thresholded_scores[thresholded_scores >= threshold] = 1
                # thresholded_scores[thresholded_scores < threshold] = 0 -> don't modify things below the threshold
                true_positives = np.sum(
                    np.count_nonzero(
                        (thresholded_scores == 1).astype(bool)
                        & (labels == 1).astype(bool)
                    )
                )
                false_positives = np.sum(
                    np.count_nonzero(
                        (thresholded_scores == 1).astype(bool)
                        & (labels == 0).astype(bool)
                    )
                )
                misses = np.sum(
                    np.count_nonzero(
                        (thresholded_scores != 1).astype(bool)
                        & (labels == 1).astype(bool)
                    )
                )
                num_passed = np.sum(np.count_nonzero(thresholded_scores == 1))
                sigmoid_threshold_to_num_passed[threshold] += num_passed
                sigmoid_threshold_to_tps_missed[threshold] += misses
                sigmoid_threshold_to_tps_passed[threshold] += true_positives
                sigmoid_threshold_to_fps_passed[threshold] += false_positives
            print(i, len(test_dataset))

        print("finished real sequences dataset, starting on decoys")
        # no padding b/c decoys are always the same length for now
        for sequence, _ in decoy_dataset:
            sequence = sequence.to(device).float()
            embeddings = model(sequence)
            scores = model.class_act(embeddings).squeeze().to("cpu").numpy()
            total_decoys += sequence.shape[0]
            thresholded_scores = scores
            for threshold in thresholds:
                thresholded_scores[thresholded_scores >= threshold] = 1
                num_decoys_passed = np.sum(np.count_nonzero(thresholded_scores == 1))
                sigmoid_threshold_to_decoys_passed[threshold] += num_decoys_passed

        fig, ax = plt.subplots(figsize=(13, 10))

        ax.plot(
            thresholds,
            [s / total_true_labels for s in sigmoid_threshold_to_tps_passed.values()],
            "ro-",
            label="percent tps recovered",
        )
        ax.plot(
            thresholds,
            [s / total_sequences for s in sigmoid_threshold_to_fps_passed.values()],
            "go-",
            label="mean fps passed per sequence",
        )
        ax.plot(
            thresholds,
            [
                1 - (s / (total_sequences * 253))
                for s in sigmoid_threshold_to_num_passed.values()
            ],
            "bo-",
            label="percent filtered",
        )
        ax1 = ax.twinx()
        ax1.plot(
            thresholds,
            [s / total_decoys for s in sigmoid_threshold_to_decoys_passed.values()],
            "ko-",
            label="decoy hits per sequence",
        )

        ax.set_title("accuracy metrics")
        ax.set_ylabel("percent of hits")
        ax1.set_ylabel("hits per sequence")
        ax.set_xlabel("sigmoid threshold")
        ax.legend()
        ax1.legend(loc="lower right")
        plt.savefig(save_fig)
        plt.close()


def main(args):
    trained_model, hparams = load_model(args.logs_dir, model_path=args.model_path)

    test_files = hparams["test_files"]

    pfam_id_to_class_code = hparams["class_code_mapping"]
    class_code_to_pfam_id = {v: k for k, v in pfam_id_to_class_code.items()}

    prot2vec = "fc1" not in hparams

    # TODO: need better ways of distinguishing b/t the two models
    test_psd = ProteinSequenceDataset(
        test_files,
        pfam_id_to_class_code,
        evaluating=True,
        use_pretrained_model_embeddings=not prot2vec,
    )

    decoys = SimpleSequenceIterator(args.decoy_path, one_hot_encode=prot2vec)

    test_collate_fn = pad_batch
    test = torch.utils.data.DataLoader(
        test_psd, batch_size=args.batch_size, shuffle=False, collate_fn=test_collate_fn
    )

    decoy_collate_fn = stack_batch

    decoys = torch.utils.data.DataLoader(
        decoys, batch_size=args.batch_size, shuffle=False, collate_fn=decoy_collate_fn
    )

    evaluation_figure_name = args.save_prefix + "_evaluated.png"
    evaluate_model(trained_model, test, decoys, evaluation_figure_name)


if __name__ == '__main__':

    log_dir = '/home/tc229954/share/prefilter/500_families_lr_ablate/4_81_1/exp_64/'

    trained_model, hparams = load_model(log_dir)
    val_files = hparams["train_files"]

    with open(os.path.join(log_dir, "new_hparams.yaml")) as src:
        cc = yaml.safe_load(src)

    pfam_id_to_class_code = cc["name_to_class_code"]
    n_classes = cc["n_classes"]

    class_code_to_pfam_id = {v: k for k, v in pfam_id_to_class_code.items()}

    psd = ProteinSequenceDataset(
        fasta_files=val_files,
        existing_name_to_label_mapping=pfam_id_to_class_code,
        evaluating=True,
        single_label=True,
        use_pretrained_model_embeddings=False
    )

    device = 'cuda'
    model = trained_model.to(device)
    dataset = torch.utils.data.DataLoader(psd,
                                          batch_size=64,
                                          collate_fn=pad_batch)

    cmat = torchmetrics.ConfusionMatrix(num_classes=n_classes).to(device)

    correct = 0
    total = len(psd)

    sequence_to_correct = {}

    with torch.no_grad():

        for features, masks, labels in dataset:
            features = features.to(device).float()
            labels = labels.to(device).argmax(dim=-1)
            preds = model.forward(features, masks).argmax(dim=-1)
            correct += torch.sum(labels == preds).item()
            cmat.update(preds, labels)

    print(correct/total)
    f1, r, p = precision_and_recall(cmat)

    print(f1, np.nanmean(r), np.nanmean(p))
