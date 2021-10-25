import os
from pytorch_lightning import seed_everything
seed_everything(1)
import torch
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin

from glob import glob
from argparse import ArgumentParser

from .models import Model, Prot2Vec
from .utils import pad_batch
from .utils import PROT_ALPHABET, pad_batch, ProteinSequenceDataset


def main(args):
    if args.schedule_lr and (args.step_lr_step_size is None or args.step_lr_decay_factor is None):
        raise ValueError('--schedule_lr requires --step_lr_step_size and --step_lr_decay_factor')
    if args.train_from_scratch:
        # TODO: add error checking to make sure at least one of the
        # arguments is filled out for prot2vec
        pass

    log_dir = args.log_dir
    data_path = args.data_path

    if '$HOME' in data_path:
        data_path = data_path.replace("$HOME", os.environ['HOME'])

    train_files = glob(os.path.join(data_path, "*train*"))
    test_files = glob(os.path.join(data_path, "*test*"))

    train = ProteinSequenceDataset(train_files,
                                   sample_sequences_based_on_family_membership=args.resample_families,
                                   sample_sequences_based_on_num_labels=args.resample_based_on_num_labels,
                                   use_pretrained_model_embeddings=not args.train_from_scratch)

    # don't resample on test.
    test = ProteinSequenceDataset(test_files,
                                  existing_name_to_label_mapping=train.name_to_class_code,
                                  use_pretrained_model_embeddings=not args.train_from_scratch)

    train.n_classes = test.n_classes
    n_classes = test.n_classes

    class_code_mapping = test.name_to_class_code

    test = torch.utils.data.DataLoader(test, batch_size=args.batch_size,
                                       shuffle=False, drop_last=False,
                                       num_workers=args.num_workers,
                                       collate_fn=pad_batch)

    train = torch.utils.data.DataLoader(train, batch_size=args.batch_size,
                                        shuffle=True, drop_last=True,
                                        num_workers=args.num_workers,
                                        collate_fn=pad_batch)

    model = Prot2Vec(learning_rate=args.learning_rate,
                     res_block_n_filters=args.res_block_n_filters,
                     vocab_size=len(PROT_ALPHABET),
                     res_block_kernel_size=args.res_block_kernel_size,
                     n_res_blocks=args.n_res_blocks,
                     res_bottleneck_factor=args.res_bottleneck_factor,
                     dilation_rate=args.dilation_rate,
                     n_classes=n_classes,
                     schedule_lr=args.schedule_lr,
                     step_lr_step_size=args.step_lr_step_size,
                     step_lr_decay_factor=args.step_lr_decay_factor,
                     test_files=test_files,
                     train_files=train_files,
                     class_code_mapping=class_code_mapping,
                     batch_size=args.batch_size,
                     pos_weight=args.pos_weight)

    save_best = pl.callbacks.model_checkpoint.ModelCheckpoint(
        monitor='val_loss',
        filename='{epoch}-{val_loss:.5f}-{val_acc:.5f}',
        save_top_k=1)

    log_lr = pl.callbacks.lr_monitor.LearningRateMonitor(logging_interval='step')

    trainer_kwargs = {
        'gpus': args.gpus,
        'num_nodes': args.num_nodes,
        'max_epochs': args.epochs,
        'check_val_every_n_epoch': args.check_val_every_n_epoch,
        'callbacks': [save_best, log_lr],
        'accelerator': 'ddp' if args.gpus else None,
        'plugins': DDPPlugin(find_unused_parameters=False),
        'precision': 16 if args.gpus else 32,
        'default_root_dir': log_dir,
        'terminate_on_nan': True
    }

    if args.tune_initial_lr:
        trainer_kwargs['auto_lr_find'] = True
        trainer = pl.Trainer(
            **trainer_kwargs
        )
        trainer.tune(model, train, test)
    else:
        trainer = pl.Trainer(
            **trainer_kwargs
        )

    trainer.fit(model, train, test)

    torch.save(model.state_dict(), os.path.join(trainer.log_dir, args.model_name))
