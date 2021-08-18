import tensorflow as tf

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
sess = tf.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
import os
import torch
import pytorch_lightning as pl

from glob import glob
from argparse import ArgumentParser

from datasets import ProteinSequenceDataset


class Model(pl.LightningModule):

    def __init__(self,
                 n_classes,
                 fc1,
                 fc2,
                 test_files,
                 train_files,
                 class_code_mapping,
                 initial_learning_rate,
                 batch_size,
                 pos_weight=1
                 ):
        super(Model, self).__init__()

        self.n_classes = n_classes
        self.initial_learning_rate = initial_learning_rate
        self.batch_size = batch_size
        self.fc1 = fc1
        self.fc2 = fc2
        self.train_files = train_files
        self.test_files = test_files
        self.class_code_mapping = class_code_mapping
        self.pos_weight = pos_weight
        self.save_hyperparameters()

        self.loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.pos_weight))
        self.class_act = torch.nn.Sigmoid()

        # 1100 because that's the dimension of the Bileschi et al model's
        # embeddings.
        self.layer_1 = torch.nn.Linear(1100, fc1)
        self.layer_2 = torch.nn.Linear(fc1, fc2)
        self.classification = torch.nn.Linear(fc2, n_classes)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.relu(self.layer_2(x))
        x = self.classification(x)
        return x

    def _loss_and_preds(self, batch):
        x, y = batch
        logits = model(x).ravel()
        labels = y.ravel()
        preds = torch.round(self.class_act(logits))
        loss = self.loss_func(logits, labels)
        acc = (torch.sum(preds == labels) / torch.numel(preds)).item()

        return loss, preds, acc

    def training_step(self, batch, batch_idx):
        loss, preds, acc = self._loss_and_preds(batch)
        self.log('loss', loss, on_step=True)
        self.log('acc', acc, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, acc = self._loss_and_preds(batch)
        self.log('val_loss', loss, on_step=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, acc = self._loss_and_preds(batch)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.initial_learning_rate)


def parser():
    ap = ArgumentParser()
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--gpus", type=int, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--layer_1_nodes", type=int, required=True)
    ap.add_argument("--layer_2_nodes", type=int, required=True)
    ap.add_argument("--normalize_output_embedding", action="store_true")
    ap.add_argument("--initial_learning_rate", type=float, required=True)
    ap.add_argument("--batch_size", type=int, required=True)
    ap.add_argument("--num_workers", type=int, required=True)
    ap.add_argument("--evaluating", action="store_true")
    ap.add_argument("--check_val_every_n_epoch", type=int, required=True)
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--pos_weight", type=float, required=True)
    return ap.parse_args()


# need to put this in inferrer

if __name__ == '__main__':
    args = parser()

    log_dir = args.log_dir
    data_path = args.data_path

    train_files = glob(os.path.join(data_path, "*train*"))
    test_files = glob(os.path.join(data_path, "*test*"))
    print(len(train_files), len(test_files))

    train = ProteinSequenceDataset(train_files)

    class_code_mapping_file = train.class_code_mapping

    test = ProteinSequenceDataset(test_files, class_code_mapping_file)

    print('--------')
    print(train.n_classes, test.n_classes)
    print('--------')

    train.n_classes = test.n_classes
    n_classes = test.n_classes

    class_code_mapping = test.name_to_class_code


    test = torch.utils.data.DataLoader(test, batch_size=args.batch_size,
                                       shuffle=False, drop_last=False)

    train = torch.utils.data.DataLoader(train, batch_size=args.batch_size,
                                        shuffle=True, drop_last=True)

    model = Model(n_classes,
                  args.layer_1_nodes,
                  args.layer_2_nodes,
                  test_files,
                  train_files,
                  class_code_mapping,
                  args.initial_learning_rate,
                  args.batch_size,
                  args.pos_weight
                  )

    save_best = pl.callbacks.model_checkpoint.ModelCheckpoint(
        monitor='val_loss',
        save_top_k=5)

    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.epochs,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        callbacks=[save_best],
        default_root_dir=log_dir,
        accelerator="ddp",
    )

    trainer.fit(model, train, test)

    torch.save(model.state_dict(), os.path.join(trainer.log_dir, args.model_name))