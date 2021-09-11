import os
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import pytorch_lightning as pl

from glob import glob
from argparse import ArgumentParser

from datasets import ProteinSequenceDataset
from prefilter.models.classification_model import Model
from prefilter.models import Prot2Vec
from prefilter.utils import pad_batch
from prefilter.utils.utils import tf_saved_model_collate_fn, PROT_ALPHABET


def parser():
    ap = ArgumentParser()
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--gpus", type=int, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--layer_1_nodes", type=int, required=True)
    ap.add_argument("--layer_2_nodes", type=int, required=True)
    ap.add_argument("--normalize_output_embedding", action="store_true")
    ap.add_argument("--learning_rate", type=float, required=True)
    ap.add_argument("--batch_size", type=int, required=True)
    ap.add_argument("--num_workers", type=int, required=True)
    ap.add_argument("--evaluating", action="store_true")
    ap.add_argument("--check_val_every_n_epoch", type=int, required=True)
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--pos_weight", type=float, required=True)
    ap.add_argument("--resample_families", action='store_true')
    ap.add_argument("--resample_based_on_num_labels", action='store_true')
    ap.add_argument("--tune_initial_lr", action='store_true')
    ap.add_argument("--schedule_lr", action='store_true')
    ap.add_argument("--step_lr_step_size", type=int, default=None)
    ap.add_argument("--step_lr_decay_factor", type=float, default=None)
    ap.add_argument("--train_from_scratch", action='store_true')
    ap.add_argument("--res_block_n_filters", type=int, default=None)
    ap.add_argument("--vocab_size", type=int, default=None)
    ap.add_argument("--res_block_kernel_size", type=int, default=None)
    ap.add_argument("--n_res_blocks", type=int, default=None)
    ap.add_argument("--res_bottleneck_factor", type=float, default=None)
    ap.add_argument("--dilation_rate", type=float, default=None)
    arguments = ap.parse_args()
    if arguments.schedule_lr and (arguments.step_lr_step_size is None or arguments.step_lr_decay_factor is None):
        ap.error('--schedule_lr requires --step_lr_step_size and --step_lr_decay_factor')
    if arguments.train_from_scratch:
        # TODO: add error checking to make sure at least one of the
        # arguments is filled out for prot2vec
        pass

    return arguments


def main(args):
    log_dir = args.log_dir
    data_path = args.data_path

    train_files = glob(os.path.join(data_path, "*train*"))
    test_files = glob(os.path.join(data_path, "*test*"))

    train = ProteinSequenceDataset(train_files,
                                   sample_sequences_based_on_family_membership=args.resample_families,
                                   sample_sequences_based_on_num_labels=args.resample_based_on_num_labels,
                                   use_pretrained_model_embeddings=not args.train_from_scratch)

    class_code_mapping_file = train.class_code_mapping

    # don't resample on test.
    test = ProteinSequenceDataset(test_files, class_code_mapping_file,
                                  use_pretrained_model_embeddings=not args.train_from_scratch)

    train.n_classes = test.n_classes
    n_classes = test.n_classes

    class_code_mapping = test.name_to_class_code
    if args.train_from_scratch:

        test = torch.utils.data.DataLoader(test, batch_size=args.batch_size,
                                           shuffle=False, drop_last=False,
                                           num_workers=args.num_workers,
                                           collate_fn=pad_batch)

        train = torch.utils.data.DataLoader(train, batch_size=args.batch_size,
                                            shuffle=True, drop_last=True,
                                            num_workers=args.num_workers,
                                            collate_fn=pad_batch)

        model = Prot2Vec(args.learning_rate,
                         args.res_block_n_filters,
                         len(PROT_ALPHABET),
                         args.res_block_kernel_size,
                         args.n_res_blocks,
                         args.res_bottleneck_factor,
                         args.dilation_rate,
                         n_classes,
                         args.schedule_lr,
                         args.step_lr_step_size,
                         args.step_lr_decay_factor,
                         test_files,
                         train_files,
                         class_code_mapping,
                         args.batch_size)

    else:

        import tensorflow as tf
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

        collate_fn = tf_saved_model_collate_fn(args.batch_size)

        test = torch.utils.data.DataLoader(test, batch_size=args.batch_size,
                                           shuffle=False, drop_last=False,
                                           num_workers=0,
                                           collate_fn=collate_fn)

        train = torch.utils.data.DataLoader(train, batch_size=args.batch_size,
                                            shuffle=True, drop_last=True,
                                            num_workers=0,
                                            collate_fn=collate_fn)
        model = Model(n_classes,
                      args.layer_1_nodes,
                      args.layer_2_nodes,
                      test_files,
                      train_files,
                      class_code_mapping,
                      args.learning_rate,
                      args.batch_size,
                      args.pos_weight,
                      args.schedule_lr,
                      args.step_lr_step_size,
                      args.step_lr_decay_factor,
                      ranking=False)

    save_best = pl.callbacks.model_checkpoint.ModelCheckpoint(
        monitor='val_loss',
        filename='{epoch}-{val_loss:.5f}',
        save_top_k=5)

    log_lr = pl.callbacks.lr_monitor.LearningRateMonitor(logging_interval='step')

    if args.tune_initial_lr:
        trainer = pl.Trainer(
            gpus=args.gpus,
            max_epochs=args.epochs,
            check_val_every_n_epoch=args.check_val_every_n_epoch,
            callbacks=[save_best, log_lr],
            default_root_dir=log_dir,
            auto_lr_find=True,
        )
        trainer.tune(model, train, test)
    else:
        trainer = pl.Trainer(
            gpus=args.gpus,
            max_epochs=args.epochs,
            check_val_every_n_epoch=args.check_val_every_n_epoch,
            callbacks=[save_best, log_lr],
            default_root_dir=log_dir,
        )

    trainer.fit(model, train, test)

    torch.save(model.state_dict(), os.path.join(trainer.log_dir, args.model_name))


if __name__ == '__main__':
    main(parser())
