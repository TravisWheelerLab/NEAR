import os
from argparse import ArgumentParser
from glob import glob

import pytorch_lightning as pl
import torch

from datasets import Word2VecStyleDataset


class Model(pl.LightningModule):

    def __init__(self,
                 embedding_dimension,
                 fc1,
                 fc2,
                 test_files,
                 train_files,
                 initial_learning_rate,
                 batch_size,
                 n_negative_samples
                 ):
        super(Model, self).__init__()

        self.embedding_dim = embedding_dimension
        self.initial_learning_rate = initial_learning_rate
        self.batch_size = batch_size
        self.n_negative_samples = n_negative_samples
        self.fc1 = fc1
        self.fc2 = fc2
        self.train_files = train_files
        self.test_files = test_files
        self.save_hyperparameters()

        self.loss_func = torch.nn.BCEWithLogitsLoss()
        self.class_act = torch.nn.Sigmoid()

        self.layer_1 = torch.nn.Linear(1100, fc1)
        self.layer_2 = torch.nn.Linear(fc1, fc2)
        self.embedding = torch.nn.Linear(fc2, embedding_dimension)

    def _get_dots(self, batch):
        targets, in_context, out_of_context = batch

        targets_embed = self.forward(targets)

        context_embed = self.forward(in_context)
        negatives_embed = self.forward(out_of_context)

        negatives_embed = torch.reshape(negatives_embed,
                                        (self.batch_size, self.n_negative_samples, self.embedding_dim))

        pos_dots = (targets_embed * context_embed).sum(axis=1).squeeze()

        targets_embed = targets_embed.transpose(-2, -1)

        neg_dots = torch.bmm(negatives_embed, targets_embed).squeeze()

        return pos_dots, neg_dots.ravel()

    def _loss_and_preds(self, batch):
        pos_dots, neg_dots = self._get_dots(batch)

        pos_dots = pos_dots.ravel()
        neg_dots = neg_dots.ravel()

        labels = torch.cat((torch.ones(pos_dots.shape[0]),
                            torch.zeros(neg_dots.shape[0]))).to('cuda')

        logits = torch.cat((pos_dots, neg_dots), axis=0)
        loss = self.loss_func(logits, labels.ravel())
        preds = torch.round(self.class_act(logits).ravel())

        acc = (torch.sum(preds == labels) / torch.numel(preds)).item()

        return loss, preds, acc

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.relu(self.layer_2(x))
        x = self.embedding(x)
        return x

    def training_step(self, batch, batch_idx):
        loss, preds, acc = self._loss_and_preds(batch)
        self.log('loss', loss, on_step=True)
        self.log('acc', loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, acc = self._loss_and_preds(batch)
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
    ap.add_argument("--embedding_dim", type=int, required=True)
    ap.add_argument("--layer_1_nodes", type=int, required=True)
    ap.add_argument("--layer_2_nodes", type=int, required=True)
    ap.add_argument("--initial_learning_rate", type=float, required=True)
    ap.add_argument("--batch_size", type=int, required=True)
    ap.add_argument("--num_workers", type=int, required=True)
    ap.add_argument("--evaluating", action="store_true")
    ap.add_argument("--check_val_every_n_epoch", type=int, required=True)
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--data_path", type=str, required=True)
    return ap.parse_args()


# need to put this in inferrer

if __name__ == '__main__':
    files = glob('../../small-dataset/json/*test*')

    args = parser()

    log_dir = args.log_dir
    root = args.data_path

    train_files = glob(os.path.join(root, "*train.json"))
    train_files = train_files[:2]
    train = Word2VecStyleDataset(json_files=train_files,
                                 max_sequence_length=None,
                                 n_negative_samples=args.n_negative_samples,
                                 evaluating=False,
                                 encoding_func=None)

    test_files = glob(os.path.join(root, "*test*.json"))
    test_files = test_files[:2]
    test = Word2VecStyleDataset(json_files=test_files,
                                max_sequence_length=None,
                                n_negative_samples=args.n_negative_samples,
                                evaluating=False,
                                encoding_func=None)

    test = torch.utils.data.DataLoader(test, batch_size=args.batch_size,
                                       shuffle=False, drop_last=True)
    train = torch.utils.data.DataLoader(train, batch_size=args.batch_size,
                                        shuffle=True, drop_last=True)

    model = Model(args.embedding_dim,
                  args.layer_1_nodes,
                  args.layer_2_nodes,
                  test_files,
                  train_files,
                  args.initial_learning_rate,
                  args.batch_size,
                  args.n_negative_samples)

    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.epochs,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        default_root_dir=log_dir,
        accelerator="ddp",
    )

    trainer.fit(model, train, test)

    torch.save(model.state_dict(), os.path.join(trainer.log_dir, args.model_name))
