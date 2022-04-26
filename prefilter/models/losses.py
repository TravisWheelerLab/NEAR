"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import os.path
import pdb
import matplotlib.pyplot as plt
import matplotlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning.losses import NTXentLoss

__all__ = ["SupConNoMasking", "SupConLoss"]


def calc_unique(x, dim=-1):
    unique, inverse = torch.unique(x, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(dim), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([dim]), perm.flip([dim])
    return unique, inverse.new_empty(unique.size(dim)).scatter_(dim, inverse, perm)


def max_pool_labels(label1, label2):
    pass


class SupConNoMasking(nn.Module):
    def __init__(self, n_conv_layers=None, device="cuda"):
        super(SupConNoMasking, self).__init__()
        if n_conv_layers is not None:
            # if it's none, assume we're using valid padding:
            # and a kernel size of three
            self.n_chop = n_conv_layers
        else:
            self.n_chop = 0
        self.supcon = SupConLoss()
        self.device = device

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, embeddings, masks, labelvecs, picture_path=None, step=None):
        """
        Need to reshape the paired embeddings into one large matrix
        """
        first_pos = False
        first_neg = False

        if picture_path is not None:
            first_pos = True
            first_neg = True

        batch_size = embeddings.shape[0] // 2

        # split the embeddings, masks, and labels into two lists of pairs
        f1, f2 = torch.split(embeddings, batch_size, dim=0)
        e1, e2 = torch.split(embeddings, batch_size, dim=0)

        f1 = f1.transpose(-1, -2)
        f2 = f2.transpose(-1, -2)
        f1 = torch.cat(torch.unbind(f1), dim=0)
        f2 = torch.cat(torch.unbind(f2), dim=0)
        f1 = torch.nn.functional.normalize(f1, dim=-1)
        f2 = torch.nn.functional.normalize(f2, dim=-1)

        x = torch.cat((f1.unsqueeze(1), f2.unsqueeze(1)), dim=1)
        loss = self.supcon(x)

        if first_pos:
            x = torch.matmul(f1, f2.T).detach().cpu()
            n_f1_valid = -1
            n_f2_valid = -1
            plt.imshow(x.float()[:n_f1_valid, :n_f2_valid], cmap="PiYG")
            plt.clim(-1, 1)
            plt.colorbar()
            os.makedirs(picture_path, exist_ok=True)
            print(f"saving to {picture_path}")
            plt.savefig(f"{picture_path}/pos_{step}.png", bbox_inches="tight")
            plt.close()

        return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        # sum over ALL of the examples (not including the self)
        # the normalization is over the entire set... positives and negatives
        # A(i) = all indices without i (without the anchor...?)
        # ok. in order to contribute nothing to the loss I can't modify
        # the denominator or the numerator.
        # so. logits in the masked positions
        #
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        # the positions that I want to mask out I'll mask out here.
        # I also need to remove those positions from the mask
        # this is the numerator; it's the cosine sims. of the positive examples.
        # let's look at
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
