import json
import os
import time
from glob import glob

import torch

import inference

__all__ = ['Word2VecStyleDataset',
           'ProteinSequenceDataset',
           ]

inferrer = inference.Inferrer(
    'trn-_cnn_random__random_sp_gpu-cnn_for_random_pfam-5356760',
    use_tqdm=False,
    batch_size=1,
    activation_type="pooled_representation"
)


class ProteinSequenceDataset(torch.utils.data.Dataset):

    def __init__(self, json_files,
                 existing_name_to_label_mapping=None,
                 evaluating=False):

        self.json_files = json_files
        self.existing_name_to_label_mapping = existing_name_to_label_mapping
        self.evaluating = evaluating

        self._build_dataset()

    def _encoding_func(self, x):

        return inferrer.get_activations([x.upper()])

    def _build_dataset(self):

        self.sequences_and_labels = []

        if self.existing_name_to_label_mapping is None:
            self.name_to_class_code = {}
            class_id = 0

        else:
            s = 'loading class mapping from file {}'.format(self.existing_name_to_label_mapping)
            print(s)

            with open(self.existing_name_to_label_mapping, 'r') as src:
                self.name_to_class_code = json.load(src)

            class_id = len(self.name_to_class_code)

        for j in self.json_files:

            with open(j, 'r') as src:
                sequence_to_labels = json.load(src)

            for sequence, labelset in sequence_to_labels.items():

                for label in labelset:

                    if label not in self.name_to_class_code:

                        if not self.evaluating:
                            self.name_to_class_code[label] = class_id
                            class_id += 1

                self.sequences_and_labels.append([labelset, sequence])

        if self.existing_name_to_label_mapping is not None and not self.evaluating:
            print('saving to existing name to label_mapping')

            with open(self.existing_name_to_label_mapping, 'w') as dst:
                json.dump(self.name_to_class_code, dst)
        else:

            self.class_code_mapping = './pfam_names_to_class_id-{}.json'.format(time.time())

            with open(self.class_code_mapping, 'w') as dst:
                json.dump(self.name_to_class_code, dst)

        self.n_classes = class_id + 1

    def __getitem__(self, key):

        if key in self.name_to_class_code:
            return self.name_to_class_code[key]
        else:
            print('this shouldn\'t happen')
            return None

    def __len__(self):

        return len(self.sequences_and_labels)

    def _make_multi_hot(self, labels):
        y = torch.zeros((self.n_classes))
        class_ids = [self.name_to_class_code[l] for l in labels]
        for idx in class_ids:
            y[idx] = 1
        return y

    def __getitem__(self, idx):

        labels, features = self.sequences_and_labels[idx][0], self.sequences_and_labels[idx][1]
        x = self._encoding_func(features)
        y = self._make_multi_hot(labels)
        return torch.tensor(x), y


if __name__ == '__main__':
    root = '../../small-dataset/json/'
    train_files = glob(os.path.join(root, "*train.json"))[:2]
    test_files = glob(os.path.join(root, "*test*"))[:2]

    train_dset = ProteinSequenceDataset(
        train_files)

    test_dset = ProteinSequenceDataset(test_files,
                                       existing_name_to_label_mapping=train_dset.class_code_mapping)
