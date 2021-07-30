import os
import pdb
import numpy as np
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor

import utils.utils as u
import models as m

try:
    from sklearn.metrics import confusion_matrix
except:
    print('cant import sklearn')

from glob import glob
from argparse import ArgumentParser


def parser():
    ap = ArgumentParser()
    ap.add_argument('--log_dir', required=True)
    ap.add_argument('--gpus', type=int, required=True)
    ap.add_argument('--epochs', type=int, required=True)
    ap.add_argument('--res_block_n_filters', type=int, required=True)
    ap.add_argument('--res_block_kernel_size', type=int, required=True)
    ap.add_argument('--vocab_size', type=int, required=True)
    ap.add_argument('--n_res_blocks', type=int, required=True)
    ap.add_argument('--res_bottleneck_factor', type=float, required=True)
    ap.add_argument('--embedding_dim', type=int, required=True)
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--normalize_output_embedding', action='store_true')
    ap.add_argument('--max_sequence_length', default=None)
    ap.add_argument('--initial_learning_rate', type=float, required=True)
    ap.add_argument('--batch_size', type=int, required=True)
    ap.add_argument('--num_workers', type=int, required=True)
    ap.add_argument('--gamma', type=float, required=True)
    ap.add_argument('--n_negative_samples', type=int, required=True)
    ap.add_argument('--evaluating', action='store_true')
    ap.add_argument('--device', type=str, required=True)
    ap.add_argument('--pooling_layer_type', type=str, required=True)
    ap.add_argument('--check_val_every_n_epoch', type=int, required=True)
    ap.add_argument('--model_name', type=str, required=True)

    return ap.parse_args()


if __name__ == '__main__':
    args = parser()

    log_dir = args.log_dir
    root = args.data_path

    train = glob(os.path.join(root, "*train.json"))
    test = glob(os.path.join(root, "*test-split.json"))

    loss_func = torch.nn.BCEWithLogitsLoss()

    test_files = test[:2]
    train_files = train[:len(train) // 2]
    valid_files = test[:2]

    model = m.Prot2Vec(
        res_block_n_filters=args.res_block_n_filters,
        vocab_size=args.vocab_size,
        pooling_layer_type=args.pooling_layer_type,
        res_block_kernel_size=args.res_block_kernel_size,
        n_res_blocks=args.n_res_blocks,
        res_bottleneck_factor=args.res_bottleneck_factor,
        embedding_dim=args.embedding_dim,
        loss_func=loss_func,
        test_files=test_files,
        train_files=train_files,
        valid_files=valid_files,
        normalize_output_embedding=args.normalize_output_embedding,
        max_sequence_length=args.max_sequence_length,
        initial_learning_rate=args.initial_learning_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        gamma=args.gamma,
        n_negative_samples=args.n_negative_samples,
        evaluating=args.evaluating,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # from argparse args could be useful
    # this could be too'use_horovod',
    trainer = Trainer(gpus=args.gpus,
                      max_epochs=args.epochs,
                      check_val_every_n_epoch=args.check_val_every_n_epoch,
                      default_root_dir=log_dir,
                      callbacks=[lr_monitor],
                      accelerator='ddp')

    trainer.fit(model)
    print('hi')

    print(os.path.join(trainer.log_dir, args.model_name))
    torch.save(model.state_dict(),
               os.path.join(trainer.log_dir, args.model_name))
