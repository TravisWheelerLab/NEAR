import pdb

import pytorch_lightning as pl
import math
import torch
import torchmetrics
import torch.nn as nn

__all__ = ['Prot2Vec']


class ResidualBlock(nn.Module):

    def __init__(self,
                 filters,
                 resnet_bottleneck_factor,
                 kernel_size,
                 layer_index,
                 first_dilated_layer,
                 dilation_rate,
                 stride=1):

        super(ResidualBlock, self).__init__()

        self.filters = filters

        shifted_layer_index = layer_index - first_dilated_layer + 1
        dilation_rate = int(max(1, dilation_rate ** shifted_layer_index))
        self.num_bottleneck_units = math.floor(
            resnet_bottleneck_factor * self.filters
        )
        self.bn1 = torch.nn.BatchNorm1d(self.filters)
        # need to pad 'same', so output has the same size as input
        # project down to a smaller number of self.filters with a larger kernel size
        self.conv1 = torch.nn.Conv1d(self.filters,
                                     self.num_bottleneck_units,
                                     kernel_size=kernel_size,
                                     dilation=dilation_rate,
                                     padding='same')

        self.bn2 = torch.nn.BatchNorm1d(self.num_bottleneck_units)
        # project back up to a larger number of self.filters w/ a kernel size of 1 (a local
        # linear transformation) No padding needed sin
        self.conv2 = torch.nn.Conv1d(self.num_bottleneck_units, self.filters, kernel_size=1,
                                     dilation=1)

    def _forward(self, x):
        features = self.bn1(x)
        features = torch.nn.functional.relu(features)
        features = self.conv1(features)
        features = self.bn2(features)
        features = torch.nn.functional.relu(features)
        features = self.conv2(features)
        return features + x

    def _masked_forward(self, x, mask):
        features = self.bn1(x)
        features = torch.nn.functional.relu(features)
        features = self.conv1(features)
        features[mask.expand(-1, self.num_bottleneck_units, -1)] = 0
        features = self.bn2(features)
        features = torch.nn.functional.relu(features)
        features = self.conv2(features)
        features[mask.expand(-1, self.filters, -1)] = 0
        return features + x

    def forward(self, x, mask=None):
        if mask is None:
            return self._forward(x)
        else:
            return self._masked_forward(x, mask)


class Prot2Vec(pl.LightningModule):
    """ 
    Convolutional network for protein family prediction.
    """

    def __init__(self,
                 learning_rate,
                 res_block_n_filters,
                 vocab_size,
                 res_block_kernel_size,
                 n_res_blocks,
                 res_bottleneck_factor,
                 dilation_rate,
                 n_classes,
                 schedule_lr,
                 step_lr_step_size,
                 step_lr_decay_factor,
                 test_files,
                 train_files,
                 class_code_mapping,
                 batch_size,
                 pos_weight,
                 normalize_output_embedding=True):

        super().__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.res_block_n_filters = res_block_n_filters
        self.vocab_size = vocab_size
        self.res_block_kernel_size = res_block_kernel_size
        self.n_res_blocks = n_res_blocks
        self.res_bottleneck_factor = res_bottleneck_factor
        self.dilation_rate = dilation_rate
        self.n_classes = n_classes
        self.schedule_lr = schedule_lr
        self.step_lr_step_size = step_lr_step_size
        self.step_lr_decay_factor = step_lr_decay_factor
        self.normalize_output_embedding = normalize_output_embedding
        self.train_files = train_files
        self.test_files = test_files
        self.class_code_mapping = class_code_mapping
        self.pos_weight = pos_weight

        self._setup_layers()

        self.loss_func = torch.nn.CrossEntropyLoss()
        self.class_act = torch.nn.Softmax(dim=-1)
        self.accuracy = torchmetrics.Accuracy()

        self.save_hyperparameters()

    def _setup_layers(self):

        self.initial_conv = nn.Conv1d(in_channels=self.vocab_size,
                                      out_channels=self.res_block_n_filters,
                                      kernel_size=self.res_block_kernel_size,
                                      padding='same')

        self.embedding_trunk = torch.nn.ModuleList()

        for layer_index in range(self.n_res_blocks):
            self.embedding_trunk.append(ResidualBlock(self.res_block_n_filters,
                                                      self.res_bottleneck_factor,
                                                      self.res_block_kernel_size,
                                                      layer_index,
                                                      1,
                                                      self.dilation_rate))

        self.classification_layer = torch.nn.Linear(self.res_block_n_filters,
                                                    self.n_classes)

    def _masked_forward(self, x, mask):
        """
        Before each convolution or batch normalization operation, we zero-out
        the features in any location that corresponds to padding in the input
        sequence 
        """
        x = self.initial_conv(x)
        # TODO: Code errors when dilation_rate is too high. The error is
        # unintelligible; figure out what's going on.
        for layer in self.embedding_trunk:
            x = layer(x, mask)
        # re-zero regions
        x[mask.expand(-1, self.res_block_n_filters, -1)] = 0
        # and do an aggregation operation
        # TODO: replace denominator of mean with the correct
        # sequence length. Also add two learnable params:
        # a power on the denominator and numerator
        return x.mean(axis=-1)

    def _forward(self, x):
        x = self.initial_conv(x)
        for layer in self.embedding_trunk:
            x = layer(x)
        return x.mean(axis=-1)

    def forward(self, x, mask=None):
        if mask is None:
            embeddings = self._forward(x)
        else:
            embeddings = self._masked_forward(x, mask)

        if self.normalize_output_embedding:
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1, p=2)

        classified = self.classification_layer(embeddings)

        return classified

    def _calc_loss(self, batch):
        features, masks, labels = batch
        logits = self.forward(features, masks)
        preds = self.class_act(logits).argmax(dim=-1)
        labels = labels.argmax(dim=-1)
        loss = self.loss_func(logits, labels)
        acc = self.accuracy(preds, labels)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._calc_loss(batch)
        return {'loss': loss, 'train_acc': acc}

    def validation_step(self, batch, batch_idx):
        loss, acc = self._calc_loss(batch)
        return {'val_loss': loss, 'val_acc': acc}

    def training_epoch_end(self, outputs):
        train_loss = self.all_gather([x['loss'] for x in outputs])
        train_acc = self.all_gather([x['train_acc'] for x in outputs])
        loss = torch.mean(torch.cat(train_loss, 0))
        acc = torch.mean(torch.cat(train_acc, 0))
        self.log('train_loss', loss)
        self.log('train_acc', acc)

    def validation_epoch_end(self, outputs):
        val_loss = self.all_gather([x['val_loss'] for x in outputs])
        val_acc = self.all_gather([x['val_acc'] for x in outputs])
        loss = torch.mean(torch.cat(val_loss, 0))
        acc = torch.mean(torch.cat(val_acc, 0))
        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        loss, acc = self._calc_loss(batch)
        return loss

    def configure_optimizers(self):
        if self.schedule_lr:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            return {'optimizer': optimizer,
                    'lr_scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_lr_step_size,
                                                                    gamma=self.step_lr_decay_factor)}
        else:
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
