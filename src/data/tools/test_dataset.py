"""Code for testing sequence data generators and loss functions"""

import os

import pytorch_lightning as pl
import torch

from src import models
from src.datasets.alignmentgenerator import (
    # AlignmentGeneratorWithIndels,
    # AlignmentGenerator,
    AlignmentGeneratorIndelsMultiPos,
)
from src.utils import pluginloader
from src.utils.losses import NpairLoss, SupConLoss


def supconloss(embeddings, labels_1=None, labels_2=None, mask=None):
    """Tester for the supervised contrastive loss"""

    embeddings_transposed = embeddings.transpose(-1, -2)
    # batch_size x sequence_length x embedding_dimension

    emb1, emb2 = torch.split(embeddings_transposed, embeddings.shape[0] // 2, dim=0,)
    emb1 = torch.cat(torch.unbind(emb1, dim=0))  # original seq embeddings
    emb2 = torch.cat(torch.unbind(emb2, dim=0))  # mutated seq embeddings
    # ((batch_size/2) * sequence_length) x embedding_dimension

    if mask is not None:
        l1_ = torch.where(~labels_1.isnan())[0]
        l2_ = torch.where(~labels_2.isnan())[0]
        emb1_ = torch.nn.functional.normalize(emb1[l1_], dim=-1)
        emb2_ = torch.nn.functional.normalize(emb2[l2_], dim=-1)
        emb1[l1_] = emb1_
        emb2[l2_] = emb2_
    else:
        emb1 = torch.nn.functional.normalize(emb1, dim=-1)
        emb2 = torch.nn.functional.normalize(emb2, dim=-1)
    loss = SupConLoss()

    return loss(torch.cat((emb1.unsqueeze(1), emb2.unsqueeze(1)), dim=1), mask=mask)


def npairsloss(embeddings, mask=None, a_indices=None, p_indices=None):
    """Tester for the npairs loss"""

    embeddings_transposed = embeddings.transpose(
        -1, -2
    )  # batch_size x sequence_length x embedding_dimension

    emb1, emb2 = torch.split(
        embeddings_transposed,
        embeddings.shape[0] // 2,
        dim=0,  # both are (batch_size /2 , sequence_length, embedding_dimension)
    )  # -- see datasets collate_fn
    emb1 = torch.cat(torch.unbind(emb1, dim=0))  # original seq embeddings
    emb2 = torch.cat(torch.unbind(emb2, dim=0))  # mutated seq embeddings
    # ((batch_size/2) * sequence_length) x embedding_dimension

    loss = NpairLoss()

    return loss(emb1, emb2, mask, a_indices, p_indices)


import pdb

if __name__ == "__main__":
    HOME = os.environ["HOME"]
    TEST_UNGAPPED = False
    TEST_GAPPED = True

    model_dict = {
        m.__name__: m for m in pluginloader.load_plugin_classes(models, pl.LightningModule)
    }

    model_class = model_dict["ResNet1dMultiPos"]
    DEVICE = "cuda"
    # device = "cpu"
    model = model_class(
        learning_rate=1e-5, log_interval=100, in_channels=20, res_block_n_filters=256,
    )
    print("Loaded model")

    ALI_PATH = "/xdisk/twheeler/daphnedemekas/train_paths-multipos.txt"

    train_dataset_ungapped = AlignmentGeneratorIndelsMultiPos(ALI_PATH, seq_len=128)
    # x = train_dataset_ungapped.__getitem__(1)
    # y = train_dataset_ungapped.__getitem__(2)

    # #each of x and y are len 2 (seq, indices)
    # #but seq and indices can be of different shape
    # pdb.set_trace()

    train_dataloader_ungapped = torch.utils.data.DataLoader(
        train_dataset_ungapped,
        collate_fn=train_dataset_ungapped.collate_fn(),
        batch_size=32,
        num_workers=6,
        drop_last=True,
    )

    dataiter_ungapped = iter(train_dataloader_ungapped)
    batch = next(dataiter_ungapped)
    pdb.set_trace()
    features_ungapped, seq1_raw, seq2_raw = next(dataiter_ungapped)
    embeddings_ungapped = model(features_ungapped)
    npairs_ungapped = npairsloss(embeddings_ungapped)

    # if TEST_UNGAPPED:
    #     ALI_PATH = "/xdisk/twheeler/daphnedemekas/train_paths2.txt"

    #     train_dataset_ungapped = AlignmentGenerator(ALI_PATH, seq_len=128)

    #     train_dataloader_ungapped = torch.utils.data.DataLoader(
    #         train_dataset_ungapped,
    #         collate_fn=train_dataset_ungapped.collate_fn(),
    #         batch_size=32,
    #         num_workers=6,
    #         drop_last=True,
    #     )

    #     dataiter_ungapped = iter(train_dataloader_ungapped)
    #     features_ungapped, seq1_raw, seq2_raw = next(dataiter_ungapped)
    #     embeddings_ungapped = model(features_ungapped)
    #     npairs_ungapped = npairsloss(embeddings_ungapped)

    #     print("Ungapped")
    #     print(npairs_ungapped)

    # if TEST_GAPPED:
    #     ALI_PATH = "/xdisk/twheeler/daphnedemekas/train_paths2.txt"
    #     train_dataset_indels = AlignmentGeneratorWithIndels(
    #         ALI_PATH, seq_len=128
    #     )

    #     train_dataloader_indels = torch.utils.data.DataLoader(
    #         train_dataset_indels,
    #         collate_fn=train_dataset_indels.collate_fn(),
    #         batch_size=32,
    #         num_workers=6,
    #         drop_last=True,
    #     )

    #     val_dataset_indels = AlignmentGeneratorWithIndels(
    #         "/xdisk/twheeler/daphnedemekas/valpaths2.txt", seq_len=128
    #     )

    #     val_dataloader_indels = torch.utils.data.DataLoader(
    #         val_dataset_indels,
    #         collate_fn=train_dataset_indels.collate_fn(),
    #         batch_size=32,
    #         num_workers=6,
    #         drop_last=True,
    #     )
    #     dataiter_indels = iter(val_dataloader_indels)
    #     for i in range(len(dataiter_indels)):
    #         (
    #             seq1,
    #             feature1_indices,
    #             seq2,
    #             feature2_indices,
    #         ) = next(dataiter_indels)
    #         print(i)

    #     while True:
    #         dataiter_indels = iter(train_dataloader_indels)

    #     (
    #         seq1,
    #         feature1_indices,
    #         seq2,
    #         feature2_indices,
    #         seq1_raw_,
    #         seq2_raw_,
    #         seq1_pure,
    #         seq2_pure,
    #     ) = next(dataiter_indels)

    #     seq_len = feature1_indices.shape[1]

    #     for batch_idx in range(feature1_indices.shape[0]):
    #         feature1_indices[batch_idx] += batch_idx * seq_len
    #         feature2_indices[batch_idx] += batch_idx * seq_len

    #     labels1 = torch.cat(torch.unbind(feature1_indices, dim=0))
    #     labels2 = torch.cat(torch.unbind(feature2_indices, dim=0))

    #     labelmat = torch.eq(labels1.unsqueeze(1), labels2.unsqueeze(0)).float()
    #     e1_indices = torch.where(~labels1.isnan())[0]
    #     e2_indices = torch.where(~labels2.isnan())[0]

    #     features_indels = torch.cat([seq1, seq2], dim=0)
    #     embeddings_indels = model(features_indels)
    #     npairs_indels = npairsloss(
    #         embeddings_indels, labelmat, e1_indices, e2_indices
    #     )

    #     print("Indels")
    #     print(npairs_indels)
