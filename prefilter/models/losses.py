"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import pdb
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning.losses import NTXentLoss

__all__ = ["SupConLoss", "AllVsAllLoss", "CustomLoss"]


class CustomLoss(nn.Module):
    def __init__(self, n_conv_layers=None, device="cuda"):
        super(CustomLoss, self).__init__()
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

    def forward(self, embeddings, embeddings_mask, labelvecs, picture=None):
        """
        Need to reshape the paired embeddings into one large matrix.
        """

        first_pos = False
        first_neg = False
        if picture is not None:
            first_pos = True
            first_neg = True

        # split the labels into two lists
        labelvecs0 = labelvecs[: embeddings.shape[0]]
        labelvecs1 = labelvecs[embeddings.shape[0] :]
        supcon_loss = 0

        for embed, embed_mask, labelvec0, labelvec1 in zip(
            embeddings, embeddings_mask, labelvecs0, labelvecs1
        ):
            # this is a pair.
            # we chop off 1 pixel on each end on each convolutional layer
            # (of which there are N).
            # So if the input is M, then output is M-2*N.
            # In our resnet, we have 18 residual layers + 1 initial conv.
            # The output is always M - 38.
            # Say our input is 126. Then our output is 88.
            # We chop off 19 pixels from each side. However, the input sequence is not always
            # the entire length of the batch. We always chop off 19 AAs from the left side,
            # and sometimes less from the right!
            # the issue is when we have sequences that are close to the length of the batch.
            # in that case, if the label vector is bigger than the embedding, we've chopped amino acids
            # off the right hand side.
            # Since we know there can only be one place the label vector is chopped, we can just
            # match the sizes after the left hand side chop.
            embed_0 = embed[0][~embed_mask[0].expand(256, -1)].view(256, -1)
            embed_1 = embed[1][~embed_mask[1].expand(256, -1)].view(256, -1)

            l0 = labelvec0[self.n_chop :]
            l1 = labelvec1[self.n_chop :]

            if l0.shape[0] > embed_0.shape[-1]:
                l0 = l0[: embed_0.shape[-1]]
            if l1.shape[0] > embed_1.shape[-1]:
                l1 = l1[: embed_1.shape[-1]]

            # Now I need to set up the pairwise matrix.
            # Nx2xK
            # where N is the number of amino acids, 2 is the pair, and K is the embedding dimension.
            # what about unaligned amino acids? I'll just not look at them for now.
            # if two AAs have the same label they're aligned. They won't be aligned in
            # the label vector, though.
            embed_0 = embed_0.transpose(0, 1)
            embed_1 = embed_1.transpose(0, 1)
            if first_pos:
                all_dots = torch.matmul(embed[0].T, embed[1])
                plt.imshow(all_dots.cpu().detach().numpy())
                plt.colorbar()
                plt.savefig(f"pos_{picture}.png", bbox_inches="tight")
                plt.close()
                first_pos = False
            if first_neg:
                all_dots = torch.matmul(embed[0].T, embeddings[-1][1])
                plt.imshow(all_dots.cpu().detach().numpy())
                plt.colorbar()
                plt.savefig(f"neg_{picture}.png", bbox_inches="tight")
                plt.close()
                first_neg = False

            mask = torch.eq(l0.unsqueeze(0), l1.unsqueeze(0).T)
            pos_batch = torch.zeros((torch.sum(mask), 2, 256)).to(self.device)

            for k, (i, j) in enumerate(zip(*torch.where(mask))):
                pos_batch[k][0] = embed_0[j]
                pos_batch[k][1] = embed_1[i]
            # need to _mask_ embeddings.
            supcon_loss += self.supcon(
                pos_batch, labels=torch.arange(pos_batch.shape[0])
            )

        return supcon_loss


class AllVsAllLoss:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        anchors,
        logos,
        anchors_mask,
        logos_mask,
        anchor_labels,
        logo_labels,
        picture=None,
    ):
        loss = 0
        if picture is not None:
            first_pos = True
            first_neg = True

        for i, pos_embed in enumerate(anchors):
            pos_len = torch.sum(~anchors_mask[i], dim=-1)
            for j, logo_embed in enumerate(logos):
                logo_len = torch.sum(~logos_mask[j], dim=-1)
                if anchor_labels[i] == logo_labels[j]:
                    all_dots = torch.matmul(pos_embed.T, logo_embed)

                    if picture is not None and first_pos:
                        plt.imshow(all_dots.cpu().detach().numpy()[:pos_len, :logo_len])
                        plt.colorbar()
                        plt.savefig(f"pos_{picture}.png", bbox_inches="tight")
                        plt.close()
                        first_pos = False

                    # with square matrices we can use np.diag
                    all_dots = torch.diag(all_dots)
                    # get CLOSE to the optimal
                    # loss will only go down if the sum of all the dots is close to n*m
                    loss += (all_dots.shape[0] - torch.sum(all_dots)) ** 2
                else:
                    # we should minimize this
                    all_dots = torch.matmul(pos_embed.T, logo_embed)
                    if picture is not None and first_neg:
                        plt.imshow(all_dots.cpu().detach().numpy()[:pos_len, :logo_len])
                        plt.colorbar()
                        plt.savefig(f"neg_{picture}.png", bbox_inches="tight")
                        plt.close()
                        first_neg = False
                    # if the dots are negative, then the loss should go down
                    loss += torch.sum(all_dots)

        return loss / torch.tensor(anchors.shape[0] ** 2)


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
