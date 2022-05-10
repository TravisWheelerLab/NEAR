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

__all__ = ["SigmoidLoss", "WeightedNTXent", "SupConNoMasking", "SupConLoss"]

import prefilter


def _save(fpath, arr):
    plt.imshow(arr.cpu())
    plt.colorbar()
    plt.title(fpath)
    # plt.savefig(fpath, bbox_inches="tight")
    plt.show()


def calc_unique(x, dim=-1):
    unique, inverse = torch.unique(x, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(dim), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([dim]), perm.flip([dim])
    return unique, inverse.new_empty(unique.size(dim)).scatter_(dim, inverse, perm)


class SigmoidLoss(nn.Module):
    def __init__(self, n_conv_layers=None, device="cuda"):
        super(SigmoidLoss, self).__init__()
        if n_conv_layers is not None:
            # if it's none, assume we're using valid padding:
            # and a kernel size of three
            self.n_chop = n_conv_layers
        else:
            self.n_chop = 0
        self.supcon = SupConLoss()
        self.device = device
        weight_mat = compute_weight_matrix(256, device=self.device, n_neighbors=20).to(
            self.device
        )
        self.weight_mat = 1 - weight_mat

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, embeddings, masks, labelvecs, picture_path=None, step=None):
        """
        Need to reshape the paired embeddings into one large matrix
        """
        first_pos = False

        if picture_path is not None:
            first_pos = True

        batch_size = embeddings.shape[0] // 2

        # split the embeddings, masks, and labels into two lists of pairs
        f1, f2 = torch.split(embeddings, batch_size, dim=0)
        # targets;

        f1 = f1.transpose(-1, -2)
        f2 = f2.transpose(-1, -2)
        f1 = torch.cat(torch.unbind(f1), dim=0)
        f2 = torch.cat(torch.unbind(f2), dim=0)
        # f1 = torch.nn.functional.normalize(f1, dim=-1)
        # f2 = torch.nn.functional.normalize(f2, dim=-1)
        loss = 0

        for e1, e2 in zip(f1, f2):
            dots = torch.matmul(e1.unsqueeze(1), e2.unsqueeze(0))
            loss += torch.nn.functional.binary_cross_entropy_with_logits(
                dots.ravel(), self.weight_mat.ravel()
            )

        if first_pos:
            x = torch.matmul(f1, f2.T).detach().cpu()
            n_f1_valid = 250
            n_f2_valid = 250
            plt.imshow(x.float()[:n_f1_valid, :n_f2_valid], cmap="PiYG")
            # plt.clim(-1, 1)
            plt.colorbar()
            os.makedirs(picture_path, exist_ok=True)
            print(f"saving to {picture_path}")
            plt.savefig(f"{picture_path}/pos_{step}.png", bbox_inches="tight")
            plt.close()

        return loss


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
            n_f1_valid = 250
            n_f2_valid = 250
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
        # _save("anchor_dot_contrast.png", anchor_dot_contrast)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # _save("logits.png", logits)
        # _save("mask.png", mask)

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # _save("mask_repeat.png", mask)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        # _save("logits_mask.png", logits_mask)
        mask = mask * logits_mask
        # _save("mask_times_lmask.png", mask)

        weight_mat = compute_weight_matrix(
            mask.shape[0], device=device, n_neighbors=20
        ).to(device)

        # remove self-contrast with logits_mask.
        exp_logits = torch.exp(logits) * logits_mask * weight_mat
        logits = logits * weight_mat
        # _save("exp_logits.png", exp_logits)

        # divide in log space by the denominator.
        # now every single position is its dot with i,j over the sum over the rows.
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # so, if i want to entirely remove the influence of neighboring amino acids
        # from the loss, I need to make the diagonal fatter.
        #
        # In that case,
        # now each position is the loss at that position.
        # all-vs-all.
        # what????
        # wait....
        # we're only propagating loss backwards for
        # positive pairs, yet comparing them to every other amino acid in the batch.
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # _save("mean_log_prob.png", mask*log_prob)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class SupConLossNeighborMask(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLossNeighborMask, self).__init__()
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

        # compute dot products.
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # numerical stab.
        logits = anchor_dot_contrast - logits_max.detach()

        # compute a thin diagonal mask.
        diagonal_mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        diagonal_mask = diagonal_mask.repeat(anchor_count, contrast_count)
        # repeat the mask so that positions 0 and N//2 are compared.

        mask = mask.repeat(anchor_count, contrast_count)
        upper = torch.triu(torch.ones((mask.shape[0], mask.shape[0])), diagonal=5)
        lower = torch.tril(torch.ones((mask.shape[0], mask.shape[0])), diagonal=-5)
        # fix the little annoying part in the middle.
        # wait, I don't even have to do that, do I?
        mask_fix = (~upper.bool() & ~lower.bool()).float()
        mask[mask_fix == 1] = 1
        # remove self-contrast cases from the mask (diagonal).
        logits_mask = mask * (~diagonal_mask.bool())
        logits_mask = ~logits_mask.bool()
        # should have a mask where the center is quite thick and
        # there are two off-diagonals. This means that the amino acids neighboring
        # any given position won't be pushed away from one another (because
        # they aren't in the denominator).
        exp_logits = torch.exp(logits) * logits_mask
        # now, do the division by the denominator:
        # divide by the exponentiated logits.
        # this serves as the "push non-neighboring things apart" part
        # of the loss function.
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # Finally, we just want to grab the loss between pairs we know are positive.
        # so we grab the aligned amino acids from each sequence.
        # This serves to maximize dot prods of the things we know are pairs and minimize
        # dps of things on the denominator (why, then, do we have the positive pairs in the denominator as well?
        # because if all dot prods go to -1, their exp. is close to 0? and log 1 == 000000000000000000000000
        # finally, remove self-contrast cases.
        diagonal_mask.fill_diagonal_(0)
        mean_log_prob_pos = (diagonal_mask * log_prob).sum(1) / diagonal_mask.sum(1)
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class WeightedNTXent(nn.Module):
    def __init__(self, n_neighbors=30, n_conv_layers=None, device="cuda"):

        super(WeightedNTXent, self).__init__()
        self.n_neighbors = n_neighbors
        self.temperature = 0.07
        if n_conv_layers is not None:
            # if it's none, assume we're using valid padding:
            # and a kernel size of three
            self.n_chop = n_conv_layers
        else:
            self.n_chop = 0
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

        f1 = f1.transpose(-1, -2)
        f2 = f2.transpose(-1, -2)
        f1 = torch.cat(torch.unbind(f1), dim=0)
        f2 = torch.cat(torch.unbind(f2), dim=0)
        f1 = torch.nn.functional.normalize(f1, dim=-1)
        f2 = torch.nn.functional.normalize(f2, dim=-1)

        all_pairs = torch.cat((f1, f2), dim=0)

        # dot all-vs-all.
        dots = torch.matmul(all_pairs, all_pairs.T)
        pair_labels = torch.eye(f1.shape[0], device=self.device)
        pair_labels = pair_labels.repeat(2, 2)
        pair_labels = pair_labels.fill_diagonal_(0)
        # now, we have raw dots.
        # take the exp;
        dots = torch.exp(dots / self.temperature).to(self.device)
        # now, weight the denominator.
        weight_mat = compute_weight_matrix(
            dots.shape[0], device=self.device, n_neighbors=self.n_neighbors
        ).to(self.device)
        # 1s where we're far away, f(distance) otherwise.
        dots = dots * weight_mat.to(self.device)
        # now, we have a weighted matrix;
        # divide by the sum;
        probs = dots / dots.sum(1, keepdim=True)
        mean_pos = (pair_labels * probs).sum(1) / (pair_labels.sum(1)).to(self.device)

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

        return mean_pos.mean()


def weight_vector(len_dot_row, center_idxs, n_neigh, device="cpu"):
    slope = 1 / n_neigh
    if not isinstance(center_idxs, list):
        center_idxs = [center_idxs]

    func_values = torch.ones(len_dot_row, device=device)
    i = 0
    for center_idx in center_idxs:
        # get beginning and end.
        begin = center_idx - n_neigh
        if begin < 0:
            begin = 0
        end = center_idx + n_neigh
        if end >= len_dot_row:
            end = len_dot_row
        # now, make the zero vector into two aranges: one
        # for the left, one for the right:
        if begin != center_idx:
            left_funcs = torch.arange(center_idx - begin, device=device)
            func_values[begin:center_idx] = slope * torch.flip(left_funcs, [0])
        if end != center_idx:
            left_funcs = torch.arange(end - center_idx, device=device)
            func_values[center_idx:end] = slope * left_funcs
        if i == 0:
            func_values[center_idx] = 0
        else:
            func_values[center_idx] = 1
        i += 1

    return func_values


def compute_weight_matrix(side_length, device="cpu", n_neighbors=20):

    wmat = torch.zeros((side_length, side_length), device=device)
    for i in range(side_length):
        x = weight_vector(side_length, [i], n_neighbors)
        wmat[i] = x
    return wmat


if __name__ == "__main__":

    embed = torch.randn((32, 256, 22))
    lfunc = SigmoidLoss(device="cpu")
    lfunc(embed, None, None)
