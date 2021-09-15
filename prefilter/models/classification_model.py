import torch
import pytorch_lightning as pl


class Model(pl.LightningModule):

    def __init__(self,
                 n_classes,
                 fc1,
                 fc2,
                 test_files,
                 train_files,
                 class_code_mapping,
                 learning_rate,
                 batch_size,
                 schedule_lr,
                 step_lr_step_size,
                 step_lr_decay_factor,
                 pos_weight,
                 ranking=True
                 ):
        super(Model, self).__init__()

        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.fc1 = fc1
        self.fc2 = fc2
        self.train_files = train_files
        self.test_files = test_files
        self.ranking = ranking
        self.class_code_mapping = class_code_mapping
        self.pos_weight = pos_weight
        self.schedule_lr = schedule_lr
        self.step_lr_step_size = step_lr_step_size
        self.step_lr_decay_factor = step_lr_decay_factor
        self.save_hyperparameters()

        self.loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.pos_weight))
        self.class_act = torch.nn.Sigmoid()
        # 1100 because that's the dimension of the Bileschi et al model's
        # embeddings.
        if self.fc2 == 0:
            self.forward_pass = torch.nn.Sequential(
                torch.nn.Linear(1100, self.n_classes),
            )
        else:
            self.forward_pass = torch.nn.Sequential(
                torch.nn.Linear(1100, fc1),
                torch.nn.ReLU(),
                torch.nn.Linear(fc1, fc2),
                torch.nn.ReLU(),
                torch.nn.Linear(fc2, self.n_classes))

    def forward(self, x, mask=None):
        return self.forward_pass(x)

    def _loss_and_preds(self, batch):
        x, y = batch
        logits = self.forward(x).ravel()
        labels = y.ravel()
        preds = torch.round(self.class_act(logits))
        loss = self.loss_func(logits, labels)
        acc = (torch.sum(preds == labels) / torch.numel(preds)).item()
        return loss, preds, acc

    def training_step(self, batch, batch_idx):
        loss, preds, acc = self._loss_and_preds(batch)
        self.log('train_loss', loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, acc = self._loss_and_preds(batch)
        self.log('val_loss', loss, on_step=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, acc = self._loss_and_preds(batch)
        return loss

    def configure_optimizers(self):
        if self.schedule_lr:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            return {'optimizer': optimizer,
                    'lr_scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_lr_step_size,
                                                                    gamma=self.step_lr_decay_factor)}
        else:
            return {'optimizer': torch.optim.Adam(self.parameters(),
                                                  lr=self.learning_rate)}
