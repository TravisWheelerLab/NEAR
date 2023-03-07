import os
import pdb

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import models
from src.datasets.alignmentgenerator import AlignmentGeneratorWithIndels
from src.datasets.datasets import AlignmentGenerator
from src.utils import pluginloader


def cross_entropy(logits, target, size_average=True):
    if size_average:
        # here we are taking the average sum
        # which will be smaller if we have indels
        # because targets will have a lot of zeros
        # so we are taking means of a lot of zeros too
        # do we want to be doing that? no probably not. we want to be ignoring those
        # otherwise u are just learning to represent whatever gets classed as zero
        # as those will have smaller
        sum = torch.sum(-target * F.log_softmax(logits, -1), -1)
        zeros = torch.where(target.sum(1) == 0)[0]
        sum[zeros] = float("nan")
        return torch.nanmean(sum)
    else:
        return torch.sum(torch.sum(-target * F.log_softmax(logits, -1), -1))


class NpairLoss(nn.Module):
    """the multi-class n-pair loss"""

    def __init__(self, l2_reg=0.02):
        super(NpairLoss, self).__init__()
        self.l2_reg = l2_reg

    def forward(self, anchor, positive, target=None, a_indices=None, p_indices=None):
        batch_size = anchor.size(0)
        if target is None:
            target = torch.eye(batch_size, dtype=torch.float32)

        logit = torch.matmul(anchor, torch.transpose(positive, 0, 1))
        loss_ce = cross_entropy(logit, target)

        if a_indices is not None:
            l2_loss = torch.sum(anchor[a_indices] ** 2) / (len(a_indices)) + torch.sum(
                positive[p_indices] ** 2
            ) / len(p_indices)
        else:
            l2_loss = (torch.sum(anchor**2) + torch.sum(positive**2)) / batch_size

        loss = loss_ce + self.l2_reg * l2_loss * 0.25
        return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, 2, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.

        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...]," "at least 3 dimensions are required"
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
        indicesr = torch.where(mask.sum(0) != 0)[0]  # there are cases such as additions in which we

        indicesc = torch.where(mask.sum(1) != 0)[0]  # there are cases such as additions in which we
        # don't want to treat the features as pos or neg, ignore entirely
        try:
            logits = logits[indicesc]
            logits = logits[:, indicesr]

            mask = mask[indicesc]
            mask = mask[:, indicesr]

            # mask-out self-contrast cases (put zeros on the diagonal)
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(indicesc.shape[0]).view(-1, 1).to(device),
                0,
            )
            mask = mask * logits_mask
        except Exception as e:
            print(e)
            pdb.set_trace()

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos

        try:
            loss = loss.view(anchor_count, batch_size).mean()
        except Exception as e:
            print(e)
            pdb.set_trace()

        return loss


def supconloss(embeddings, l1=None, l2=None, mask=None):

    # batch_size x sequence_length x embedding_dimension
    # 32x768x200

    embeddings_transposed = embeddings.transpose(
        -1, -2
    )  # batch_size x sequence_length x embedding_dimension

    e1, e2 = torch.split(
        embeddings_transposed,
        embeddings.shape[0] // 2,
        dim=0,
    )  # both are (batch_size /2 , sequence_length, embedding_dimension))  # -- see datasets collate_fn
    e1 = torch.cat(torch.unbind(e1, dim=0))  # original seq embeddings
    e2 = torch.cat(torch.unbind(e2, dim=0))  # mutated seq embeddings
    # ((batch_size/2) * sequence_length) x embedding_dimension

    if mask is not None:
        l1_ = torch.where(~l1.isnan())[0]
        l2_ = torch.where(~l2.isnan())[0]
        e1_ = torch.nn.functional.normalize(e1[l1_], dim=-1)
        e2_ = torch.nn.functional.normalize(e2[l2_], dim=-1)
        e1[l1_] = e1_
        e2[l2_] = e2_
    else:
        e1 = torch.nn.functional.normalize(e1, dim=-1)
        e2 = torch.nn.functional.normalize(e2, dim=-1)
    loss = SupConLoss()

    l = loss(torch.cat((e1.unsqueeze(1), e2.unsqueeze(1)), dim=1), mask=mask)

    return l


def npairsloss(embeddings, mask=None, a_indices=None, p_indices=None):

    embeddings_transposed = embeddings.transpose(
        -1, -2
    )  # batch_size x sequence_length x embedding_dimension

    e1, e2 = torch.split(
        embeddings_transposed,
        embeddings.shape[0] // 2,
        dim=0,  # both are (batch_size /2 , sequence_length, embedding_dimension)
    )  # -- see datasets collate_fn
    e1 = torch.cat(torch.unbind(e1, dim=0))  # original seq embeddings
    e2 = torch.cat(torch.unbind(e2, dim=0))  # mutated seq embeddings
    # ((batch_size/2) * sequence_length) x embedding_dimension

    # e1 = torch.nn.functional.normalize(e1, dim=-1)
    # e2 = torch.nn.functional.normalize(e2, dim=-1)

    loss = NpairLoss()

    l = loss(e1, e2, mask, a_indices, p_indices)
    return l


HOME = os.environ["HOME"]

# train_dataset_ungapped = AlignmentGenerator(ali_path = "/xdisk/twheeler/daphnedemekas/train_paths2.txt",seq_len = 128)

# train_dataloader_ungapped = torch.utils.data.DataLoader(
#     train_dataset_ungapped,
#     collate_fn=train_dataset_ungapped.collate_fn(),
#     batch_size = 32,
#     num_workers = 6,
#     drop_last = True
# )

# dataiter_ungapped = iter(train_dataloader_ungapped)
# features_ungapped, seq1_raw, seq2_raw = next(dataiter_ungapped)


train_dataset_indels = AlignmentGeneratorWithIndels(
    ali_path="/xdisk/twheeler/daphnedemekas/train_paths2.txt", seq_len=128
)

train_dataloader_indels = torch.utils.data.DataLoader(
    train_dataset_indels,
    collate_fn=train_dataset_indels.collate_fn(),
    batch_size=32,
    num_workers=6,
    drop_last=True,
)
# for idx in range(100):
#     print(idx)
#     seq1, feature1_indices, seq2, feature2_indices = train_dataset_indels.__getitem__(idx)

dataiter_indels = iter(train_dataloader_indels)
seq1, feature1_indices, seq2, feature2_indices, seq1_raw_, seq2_raw_, seq1_pure, seq2_pure = next(
    dataiter_indels
)

seq_len = feature1_indices.shape[1]

for batch_idx in range(feature1_indices.shape[0]):
    feature1_indices[batch_idx] += batch_idx * seq_len
    feature2_indices[batch_idx] += batch_idx * seq_len

l1 = torch.cat(torch.unbind(feature1_indices, dim=0))
l2 = torch.cat(torch.unbind(feature2_indices, dim=0))

labelmat = torch.eq(l1.unsqueeze(1), l2.unsqueeze(0)).float()
e1_indices = torch.where(~l1.isnan())[0]
e2_indices = torch.where(~l2.isnan())[0]
pdb.set_trace()


model_dict = {m.__name__: m for m in pluginloader.load_plugin_classes(models, pl.LightningModule)}

model_class = model_dict["ResNet1d"]
device = "cuda"
# device = "cpu"
model = model_class(learning_rate=1e-5, log_interval=100, in_channels=20, res_block_n_filters=256)
print("Loaded model")
# features = torch.cat([seq1, seq2], dim = 0)

embeddings_ungapped = model(features_ungapped)

npairs_ungapped = npairsloss(embeddings_ungapped)

print("Ungapped")
print(npairs_ungapped)


features_indels = torch.cat([seq1, seq2], dim=0)
embeddings_indels = model(features_indels)
npairs_indels = npairsloss(embeddings_indels, labelmat, e1_indices, e2_indices)

print("Indels")
print(npairs_indels)
pdb.set_trace()


# seq1, seq1_indices, seq2, seq2_indices = train_dataset.__getitem__(idx)

# while not any(seq1_indices.isnan()) and not any(seq2_indices.isnan()):
#     idx += 1
#     print(seq1_indices)
#     seq1, seq1_indices, seq2, seq2_indices = train_dataset.__getitem__(idx)
# seq1 = seq1.unsqueeze(0)
# seq2 = seq2.unsqueeze(0)
# seq1_indices = seq1_indices.unsqueeze(0)
# seq2_indices = seq2_indices.unsqueeze(0)
model_dict = {m.__name__: m for m in pluginloader.load_plugin_classes(models, pl.LightningModule)}

model_class = model_dict["ResNet1d"]
device = "cpu"

model = model_class(learning_rate=1e-5, log_interval=100, in_channels=20, res_block_n_filters=256)
print("Loaded model")
features = torch.cat([seq1, seq2], dim=0)

embeddings = model(features)

feature1_indices = seq1_indices.clone()
feature2_indices = seq2_indices.clone()

seq_len = feature1_indices.shape[1]

for batch_idx in range(feature1_indices.shape[0]):
    feature1_indices[batch_idx] += batch_idx * seq_len
    feature2_indices[batch_idx] += batch_idx * seq_len
# feature1_indices = feature1_indices.contiguous().view(-1,1)
# feature2_indices = feature2_indices.contiguous().view(-1,1)
# mask = torch.eq(feature1_indices, feature2_indices.T)

embeddings_transposed = embeddings.transpose(
    -1, -2
)  # batch_size x sequence_length x embedding_dimension

e1, e2 = torch.split(
    embeddings_transposed,
    embeddings.shape[0] // 2,
    dim=0,
)  # both are (batch_size /2 , sequence_length, embedding_dimension))  # -- see datasets collate_fn
e1 = torch.cat(torch.unbind(e1, dim=0))  # original seq embeddings
e2 = torch.cat(torch.unbind(e2, dim=0))  # mutated seq embeddings
# ((batch_size/2) * sequence_length) x embedding_dimension

e1 = torch.nn.functional.normalize(e1, dim=-1)
e2 = torch.nn.functional.normalize(e2, dim=-1)

# l1 = torch.cat(torch.unbind(feature1_indices, dim=0))
# l2 = torch.cat(torch.unbind(feature2_indices, dim=0))
l1 = feature1_indices
l2 = feature2_indices

l1_insertions = l1.isnan()
l2_insertions = l2.isnan()

for idx in range(l1.shape[0]):
    length = torch.min(len(torch.where(~l1_insertions[idx])), len(torch.where(~l2_insertions[idx])))

pdb.set_trace()
first = torch.min(l1[0], l2[0])


batch_size = anchor.size(0)
target = target.view(target.size(0), 1)

target = (target == torch.transpose(target, 0, 1)).float()
target = target / torch.sum(target, dim=1, keepdim=True).float()

logit = torch.matmul(anchor, torch.transpose(positive, 0, 1))
loss_ce = cross_entropy(logit, target)
l2_loss = torch.sum(anchor**2) / batch_size + torch.sum(positive**2) / batch_size

loss = loss_ce + self.l2_reg * l2_loss * 0.25

pdb.set_trace()


# labelmat = torch.eq(l1.unsqueeze(1), l2.unsqueeze(0)).float()

# embeddings = torch.cat((e1.unsqueeze(1), e2.unsqueeze(1)), dim=1)
# batch_size = embeddings.shape[0]

# mask = labelmat


# mask = torch.eye(batch_size, dtype=torch.float32).to(device)

# contrast_count = embeddings.shape[1]
# contrast_feature = torch.cat(torch.unbind(embeddings, dim=1), dim=0) #add both amino embeddings into one tensor [batchsize*2, embedding_dimension]

# anchor_feature = contrast_feature
# anchor_count = contrast_count

# # compute logits
# anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), 0.07)
# # for numerical stability
# logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
# logits = anchor_dot_contrast - logits_max.detach()
# pdb.set_trace()
# indices = torch.where(mask.sum(1) != 0)
# logits = logits[indices]
# dim = mask.shape[0]
# # tile mask
# zero_mask = torch.eye(dim, dim).repeat(anchor_count, contrast_count)
# zero_mask[dim:,:dim] = mask
# zero_mask[:dim,dim:] = mask
# mask = zero_mask.to(device)


# #mask = mask.repeat(anchor_count, contrast_count)


# # mask-out self-contrast cases
# logits_mask = torch.scatter(torch.ones_like(mask[indices]),1,torch.arange(len(indices[0])).view(-1, 1).to(device),0,)
# mask = mask * logits_mask

# # compute log_prob
# exp_logits = torch.exp(logits) * logits_mask
# log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

# # compute mean of log-likelihood over positive
# mean_log_prob_pos = (mask[indices] * log_prob[indices]).sum(1) / mask[indices].sum(1)
# # loss
# loss = -(0.07 / 0.07) * mean_log_prob_pos

# loss = loss.view(anchor_count, int(len(indices[0])/2)).nanmean()

# pdb.set_trace()


# feature1_indices = torch.stack(seq1_indices).T
# feature2_indices = torch.stack(seq2_indices).T

# seq_len = feature1_indices.shape[1]

# for batch_idx in range(feature1_indices.shape[0]):
#     feature1_indices[batch_idx] += batch_idx * seq_len
#     feature2_indices[batch_idx] += batch_idx * seq_len

# feature1_indices = feature1_indices.contiguous().view(-1,1)
# feature2_indices = feature2_indices.contiguous().view(-1,1)
# mask = torch.eq(feature1_indices, feature2_indices.T)
# features = torch.cat([seq1, seq2], dim = 0)
# mask = mask.repeat(2, 2)
# pdb.set_trace()
