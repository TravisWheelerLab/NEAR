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

def setup_parser():
    ap = ArgumentParser()

    model_group = ap.add_mutually_exclusive_group(required=True)

    model_group.add_argument('--deepfam', action='store_true')
    model_group.add_argument('--deepnog', action='store_true')
    model_group.add_argument('--attn', action='store_true')
    model_group.add_argument('--protcnn', action='store_true')

    label_group = ap.add_mutually_exclusive_group(required=True)

    label_group.add_argument('--multilabel', action='store_true',
            help='sigmoid activation with N_CLASSES nodes on the last layer,\
            useful for doing multi-label classification')

    label_group.add_argument('--multiclass',
            type=int, help='multiclass classification. Each sequence classified\ (w/ softmax activation) into one of N_CLASSES')

    ap.add_argument('--epochs', type=int, default=10,
            help='max. number of epochs to train')
    ap.add_argument('--batch-size', required=False, default=16, type=int)
    ap.add_argument('--max-sequence-length', required=False, default=256,
            type=int, help='size to which sequences will be truncated or padded')
    ap.add_argument('--num-workers', required=False, default=4,
            type=int, help='number of workers to use when loading data')
    ap.add_argument('--encode-as-image', required=False, action='store_true',
            help='whether or not to encode residues as one-hot vector ')
    # TODO: fix encode-as-image argument to encode-as-one-hot

    ap.add_argument('--data-path', type=str, required=True, help='where the\
                    data is stored. Requires structure to be <data-path>/<test, train, val>.json')
    ap.add_argument('--lr', type=float, default=1e-3, 
            help='initial learning rate')

    ap.add_argument('--model-dir', type=str, required=True,
                     help='where to save trained models')

    ap.add_argument('--n-gpus', type=int, required=True, help='number of gpus to use')

    ap.add_argument('--model-name', type=str, required=True, help='the name of\
            the model you want to train')

    ap.add_argument('--threshold-curve', action='store_true',
            help='whether or not to save precision and recall on test set\
                    as a function of different probability thresholds. Only\
                    works if you have matplotlib installed')
    ap.add_argument('--log-freq', type=int, default=2,
                    help='when to log the threshold_curve graph')
    ap.add_argument('--n-classes', type=int, default=u.N_CLASSES, )
    ap.add_argument('--log-dir', type=str, default=None,
            help='where to save tensorboard logs')
    ap.add_argument('--tune-batch-size', action='store_true',
            help='use pt-lightning to get the largest batch size')
    ap.add_argument('--tune-initial-lr', action='store_true')
            help='use pt-lightning to get a guess for a good initial learning rate')

    loss_group = ap.add_mutually_exclusive_group(required=True)

    loss_group.add_argument('--focal-loss', action='store_true', help='whether or not \
            to use focal loss, defined in losses.py')
    loss_group.add_argument('--bce-loss', action='store_true', help='whether or not \
            to use regular BCE loss')
    loss_group.add_argument('--xent-loss', action='store_true', help='whether or not \
            to use categorical xent')

    ap.add_argument('--gamma', type=float, default=0.2, help='factor to decay lr by every\
            step-size epochs')
    ap.add_argument('--step-size', type=float, default=2, help='lr is decayed by gamma\
            every step-size epochs, in a staircase fashion')


    args = ap.parse_args()

    return args


if __name__ == '__main__':

    args = setup_parser()

    data_root = args.data_path
    batch_size = args.batch_size
    num_epochs = args.epochs
    max_sequence_length = args.max_sequence_length
    multilabel = args.multilabel
    multiclass = args.multiclass
    num_workers = args.num_workers
    n_epochs = args.epochs
    encode_as_image = args.encode_as_image
    threshold_curve = args.threshold_curve
    log_dir = args.log_dir
    log_freq = args.log_freq
    step_size = args.step_size
    gamma = args.gamma
    n_gpus = args.n_gpus
    n_classes = args.n_classes
    tune_batch_size = args.tune_batch_size
    tune_initial_lr = args.tune_initial_lr

    focal_loss = args.focal_loss
    bce_loss = args.bce_loss
    xent_loss = args.xent_loss
    model_name_suffix = args.model_name
    init_lr = args.lr

    model_dir = args.model_dir

    test = glob(os.path.join(data_root, '*test*'))
    train = glob(os.path.join(data_root, '*train*'))
    valid = glob(os.path.join(data_root, '*val*'))


    if focal_loss:
        loss_func = l.FocalLoss()
    elif bce_loss: 
        loss_func = torch.nn.BCEWithLogitsLoss()
    elif xent_loss: 
        loss_func = torch.nn.CategoricalCrossEntropyWithLogits()
    else:
        pass

    arg_dict = vars(args)

    arg_dict['metrics'] = m.configure_metrics()
    arg_dict['loss_func'] = loss_func
    arg_dict['test_files'] = test
    arg_dict['train_files'] = train
    arg_dict['valid_files'] = valid

    if args.deepfam:

        conf = m.DEEPFAM_CONFIG
        conf['n_classes'] = n_classes
        model = m.DeepFam(conf)
        model = m.ClassificationTask(model, arg_dict)
        model_name = 'deepfam{}.h5'

    elif args.deepnog:

        conf = m.DEEPNOG_CONFIG
        conf['n_classes'] = n_classes
        model = m.DeepNOG(conf)
        model = m.ClassificationTask(model, arg_dict)
        model_name = 'deepnog{}.h5'

    elif args.attn:

        conf = m.ATTN_CONFIG
        conf['n_classes'] = n_classes
        model = m.AttentionModel(conf)
        model = m.ClassificationTask(model, arg_dict)
        model_name = 'attn{}.h5'

    elif args.protcnn:

        conf = m.PROTCNN_CONFIG
        conf['n_classes'] = n_classes
        model = m.ProtCNN(conf)
        model = m.ClassificationTask(model, arg_dict)
        model_name = 'protcnn{}.h5'

    else:

        raise ValueError('one of <deepnog, deepfam, attn> required as\
                command-line-arg')

    unique_time = str(int(time.time()))
    model_name = model_name.format(unique_time) + "_" + model_name_suffix
    model_name = os.path.join(model_dir, model_name)

    if log_dir is not None:
        log_dir = os.path.join(os.getcwd(), 'lightning_logs', log_dir)

    
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

    if n_gpus > 1:
        trainer = pl.Trainer(gpus=[i for i in range(n_gpus)],
                max_epochs=num_epochs, accelerator='ddp', default_root_dir=log_dir,
                callbacks=[lr_monitor])
    else:
        trainer = pl.Trainer(gpus=1, max_epochs=num_epochs, default_root_dir=log_dir,
                callbacks=[lr_monitor])

    if tune_batch_size:

        tuner = pl.tuner.tuning.Tuner(trainer)
        new_batch_size = tuner.scale_batch_size(model)
        model.hparams.batch_size = new_batch_size
        print('Tuned batch size to {}'.format(new_batch_size))

    if tune_initial_lr:

        lr_finder = trainer.tuner.lr_find(model)
        new_learning_rate = lr_finder.suggestion()
        model.hparams.lr = new_learning_rate
        print('Tuned learning rate to {}'.format(new_learning_rate))


    trainer.fit(model)

    results = trainer.test(model)

    p = results['Precision']
    r = results['Recall']
    model_name += '{:.3f}_{:.3f}'.format(p, r)

    torch.save(model.state_dict(), model_name)
