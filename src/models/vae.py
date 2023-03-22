import pdb

import matplotlib.pyplot as plt
import torch

from src.models.sequence_vae import SequenceVAEWithIndels


class VAEIndels(SequenceVAEWithIndels):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def image_log(self, recon, concat_features, all_dots, labelmat):
        e1 = torch.cat(
            torch.unbind(torch.nn.functional.softmax(recon, dim=1), dim=-1)
        )[:200]
        e2 = torch.cat(torch.unbind(concat_features, dim=-1))[:200]

        with torch.no_grad():
            fig, ax = plt.subplots(ncols=2)
            if self.apply_contrastive_loss:
                acc = (
                    torch.round(torch.sigmoid(all_dots)) == labelmat
                ).sum() / labelmat.numel()
                ax[0].set_title(f"accuracy: {acc.item():.5f}")
            ax[0].imshow(
                e1.to("cpu").numpy().astype(float),
                interpolation="nearest",
            )
            ax[1].imshow(
                e2.to("cpu").numpy().astype(float), interpolation="nearest"
            )
            self.logger.experiment.add_figure(
                f"image", plt.gcf(), global_step=self.global_step
            )

    def _shared_step(self, batch):
        (
            seq,
            labels1,
            mutated_seq,
            labels2,
        ) = batch  # 32 pairs of sequences, each amino has a label
        concat_features = torch.cat((seq, mutated_seq), dim=0)
        embeddings, recon = self.forward(concat_features)

        # cross entropy loss
        #        loss = self.xent(recon, concat_features)
        #        loss += self.KLD

        # TODO: supervised contrastive loss
        recon_original, recon_mutated = torch.split(
            recon, concat_features.shape[0] // 2
        )
        original_features = torch.cat(torch.unbind(seq, dim=-1))
        recon_features = torch.cat(torch.unbind(recon_mutated, dim=-1))
        recon_normalized = torch.nn.functional.softmax(recon_features, dim=0)
        l1 = torch.cat(torch.unbind(labels1, dim=0))
        l2 = torch.cat(torch.unbind(labels2, dim=0))

        labelmat = torch.eq(l1.unsqueeze(1), l2.unsqueeze(0)).float()
        all_dots = torch.matmul(original_features, recon_normalized.T)
        # loss = torch.nn.functional.binary_cross_entropy_with_logits(
        #      all_dots.ravel(), labelmat.ravel()
        # )

        # contrastive loss on embeddings
        embeddings_original, embeddings_mutated = torch.split(
            embeddings, embeddings.shape[0] // 2
        )

        embeddings_original = torch.cat(
            torch.unbind(embeddings_original, dim=-1), dim=-1
        )
        embeddings_mutated = torch.cat(
            torch.unbind(embeddings_mutated, dim=-1), dim=-1
        )

        embeddings_original = torch.nn.functional.normalize(
            embeddings_original, dim=-1
        )
        embeddings_mutated = torch.nn.functional.normalize(
            embeddings_mutated, dim=-1
        )

        # fmt: off
        loss = self.supcon(torch.cat((embeddings_mutated.unsqueeze(1), embeddings_original.unsqueeze(1)), dim=1))


        if self.global_step % self.log_interval == 0:
            self.image_log(recon, concat_features, all_dots, labelmat)
        
        
        return loss
