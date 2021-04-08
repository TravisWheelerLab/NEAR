import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import time
import pdb
import numpy as np
import torch
import pytorch_lightning as pl

import data.utils as u
import models as m
import losses as l

try:
    from sklearn.metrics import confusion_matrix
except:
    print('cant import sklearn')

from pytorch_lightning.metrics import MetricCollection, Accuracy, Precision, Recall
from glob import glob
from argparse import ArgumentParser


n_classes = 17646 


if __name__ == '__main__':

    ap = ArgumentParser()

    model_group = ap.add_mutually_exclusive_group(required=True)

    model_group.add_argument('--deepfam', action='store_true')
    model_group.add_argument('--deepnog', action='store_true')
    model_group.add_argument('--attn', action='store_true')

    label_group = ap.add_mutually_exclusive_group(required=True)
    label_group.add_argument('--binary-multilabel', action='store_true',
            help='sigmoid activation with N_CLASSES nodes on the last layer,\
            useful for doing multi-label classification')

    label_group.add_argument('--multiclass',
            type=int, help='multiclass classification. Each sequence classified\
            (w/ softmax activation) into one of N_CLASSES')

    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-size', required=False, default=16, type=int)
    ap.add_argument('--max-sequence-length', required=False, default=256,
            type=int, help='size to which sequences will be truncated or padded')
    ap.add_argument('--num-workers', required=False, default=4,
            type=int, help='number of workers to use when loading data')
    ap.add_argument('--encode-as-image', required=False, action='store_true')

    ap.add_argument('--data-path', type=str, required=True, help='where the\
                    data is stored, in structure of <data-path>/<test, train, val>')
    ap.add_argument('--lr', type=float, default=1e-3, help='learning rate')

    ap.add_argument('--model-dir', type=str, required=True, help='where to save\
    trained models')

    ap.add_argument('--model-name', type=str, required=True, help='the name of\
            the model you want to train')

    loss_group = ap.add_mutually_exclusive_group(required=True)
    loss_group.add_argument('--focal-loss', action='store_true', help='whether or not \
            to use focal loss, defined in losses.py')
    loss_group.add_argument('--bce-loss', action='store_true', help='whether or not \
            to use regular BCE loss')
    loss_group.add_argument('--xent-loss', action='store_true', help='whether or not \
            to use categorical xent')

    args = ap.parse_args()

    data_root = args.data_path
    batch_size = args.batch_size
    num_epochs = args.epochs
    max_sequence_length = args.max_sequence_length
    binary_multilabel = args.binary_multilabel
    multiclass = args.multiclass
    num_workers = args.num_workers
    n_epochs = args.epochs
    encode_as_image = args.encode_as_image

    focal_loss = args.focal_loss
    bce_loss = args.bce_loss
    xent_loss = args.xent_loss

    model_name_suffix = args.model_name
    init_lr = args.lr

    model_dir = args.model_dir


    test = glob(os.path.join(data_root, '*test*'))
    train = glob(os.path.join(data_root, '*train*'))
    valid = glob(os.path.join(data_root, '*val*'))

    test = u.ProteinSequenceDataset(test,
                              max_sequence_length,
                              encode_as_image,
                              u.N_CLASSES,
                              binary_multilabel)

    train = u.ProteinSequenceDataset(train,
                               max_sequence_length,
                               encode_as_image,
                               u.N_CLASSES,
                               binary_multilabel)

    validation  = u.ProteinSequenceDataset(valid,
                                     max_sequence_length,
                                     encode_as_image,
                                     u.N_CLASSES,
                                     binary_multilabel)

    train = torch.utils.data.DataLoader(train, batch_size=batch_size,
            num_workers=num_workers)
    test = torch.utils.data.DataLoader(test, batch_size=batch_size, 
            num_workers=num_workers)
    valid = torch.utils.data.DataLoader(validation, batch_size=batch_size,
            num_workers=num_workers)

    loss_func = None
    if focal_loss:
        loss_func = l.FocalLoss()
    elif bce_loss: 
        loss_func = torch.nn.BCEWithLogitsLoss()
    elif xent_loss: 
        loss_func = torch.nn.CategoricalCrossEntropyWithLogits()
    else:
        pass


    if args.deepfam:

        deepfam_config = {
                'n_classes':u.N_CLASSES,
                'kernel_size': [8, 12, 16, 20, 24, 28, 32, 36],
                'n_filters': 150,
                'dropout': 0.3,
                'vocab_size': 23,
                'hidden_units': 2000,
                'multilabel_classification': binary_multilabel,
                'lr':init_lr,
                'alphabet_size':len(u.PROT_ALPHABET),
                'optim':torch.optim.Adam,
                'loss_func':loss_func,
                'metrics':m.configure_metrics()
                }

        model = m.ClassificationTask(m.DeepFam(deepfam_config), deepfam_config)
        model_name = 'deepfam{}.h5'

    elif args.deepnog:

        deepnog_config = {
                'n_classes':u.N_CLASSES,
                'kernel_size': [8, 12, 16, 20, 24, 28, 32, 36],
                'encoding_dim':len(u.PROT_ALPHABET),
                'n_filters': 150,
                'dropout': 0.3,
                'pooling_layer_type':'avg',
                'vocab_size': 23,
                'hidden_units': 2000,
                'multilabel_classification': binary_multilabel,
                'lr':init_lr,
                'alphabet_size':len(u.PROT_ALPHABET),
                'optim':torch.optim.Adam,
                'loss_func':loss_func,
                'metrics':m.configure_metrics()
                }

        model = m.ClassificationTask(m.DeepNOG(deepnog_config), deepnog_config)
        model_name = 'deepnog{}.h5'

    elif args.attn:

        attn_config = {
                'n_classes':u.N_CLASSES,
                'kernel_size': [8, 12, 16, 20, 24, 28, 32, 36],
                'encoding_dim':len(u.PROT_ALPHABET),
                'n_filters': 150,
                'dropout': 0.3,
                'pooling_layer_type':'avg',
                'qkv_embed_dim': 16,
                'hidden_units': 200,
                'multilabel_classification': binary_multilabel,
                'alphabet_size':len(u.PROT_ALPHABET),
                'lr':init_lr,
                'optim':torch.optim.Adam,
                'loss_func':loss_func,
                'metrics':m.configure_metrics(),
                'mha_embed_dim':32,
                'num_heads':2,
                }

        model = m.ClassificationTask(m.AttentionModel(attn_config), attn_config)
        model_name = 'attn{}.h5'

    else:

        raise ValueError('one of <deepnog, deepfam, attn> required as\
                command-line-arg')

    unique_time = str(int(time.time()))
    model_name = model_name.format(unique_time) + "_" + model_name_suffix

    trainer = pl.Trainer(gpus=1, max_epochs=num_epochs)

    trainer.fit(model, train, valid)
    model_name = os.path.join(model_dir, model_name)

    trainer.test(model, test)

    torch.save(model, model_name)
