import torch
import os
from torch.nn import functional as F
from argparse import ArgumentParser
import pytorch_lightning as pl
from pl_examples.basic_examples.mnist_datamodule import MNISTDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


class LitClassifier(pl.LightningModule):
    def __init__(self, hidden_dim: int = 128, learning_rate: float = 0.0001):
        super().__init__()
        self.save_hyperparameters()

        self.l1 = torch.nn.Linear(28 * 28, self.hparams.hidden_dim)
        self.l2 = torch.nn.Linear(self.hparams.hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        probs = self(x)
        # we currently return the accuracy as the validation_step/test_step is run on the IPU devices.
        # Outputs from the step functions are sent to the host device, where we calculate the metrics in
        # validation_epoch_end and test_epoch_end for the test_step.
        acc = self.accuracy(probs, y)
        return acc

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = self.accuracy(logits, y)
        return acc

    def accuracy(self, logits, y):
        # currently IPU poptorch doesn't implicit convert bools to tensor
        # hence we use an explicit calculation for accuracy here. Once fixed in poptorch
        # we can use the accuracy metric.
        acc = torch.sum(torch.eq(torch.argmax(logits, -1), y).to(torch.float32)) / len(
            y
        )
        return acc

    def validation_epoch_end(self, outputs) -> None:
        # since the training step/validation step and test step are run on the IPU device
        # we must log the average loss outside the step functions.
        self.log("val_acc", torch.stack(outputs).mean(), prog_bar=True)

    def test_epoch_end(self, outputs) -> None:
        self.log("test_acc", torch.stack(outputs).mean())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


if __name__ == "__main__":

    ap = ArgumentParser()
    ap.add_argument("--experiment_dir")
    ap.add_argument("--max_iter", type=int, required=True)
    ap.add_argument("--load_from_ckpt", action="store_true")
    ap.add_argument("--learning_rate", type=float)

    args = ap.parse_args()

    dm = MNISTDataModule(batch_size=1024)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.experiment_dir, "checkpoints"),
        filename="ckpt-{epoch}",
        save_top_k=-1,
        verbose=True,
    )

    model = LitClassifier(learning_rate=args.learning_rate)
    last_epoch = 0

    if args.load_from_ckpt:
        from glob import glob

        checkpoints = sorted(
            glob(os.path.join(args.experiment_dir, "checkpoints", "*pt"))
        )
        stripped_ckpt = list(
            map(lambda x: os.path.splitext(os.path.basename(x))[0], checkpoints)
        )
        ckpt_path = checkpoints[-1]
        last_epoch = int(stripped_ckpt[-1].split("=")[-1]) + 1
        model = model.load_from_checkpoint(ckpt_path, map_location=torch.device("cuda"))

    logger = TensorBoardLogger(args.experiment_dir, name="", version="")
    min_unit = 4  # epochs

    trainer = pl.Trainer(
        max_epochs=(last_epoch + args.max_iter) * min_unit,
        logger=logger,
        callbacks=[checkpoint_callback],
        resume_from_checkpoint=ckpt_path if args.load_from_ckpt else None,
        log_every_n_steps=1,
        gpus=1,
    )

    trainer.fit(model, datamodule=dm)

    results = trainer.test(model, datamodule=dm)[0]
    # >>> print(results)
    # {'test_acc': 0.4504123330116272}
    print(results)

    with open(os.path.join(args.experiment_dir, "checkpoint.txt"), "w") as dst:
        dst.write(f"test_acc:{results['test_acc']}")
