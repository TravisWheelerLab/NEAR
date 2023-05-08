import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy(logits, target, size_average=True):
    if size_average:
        sum = torch.sum(-target * F.log_softmax(logits, -1), -1)
        zeros = torch.where(target.sum(1) == 0)[0]
        sum[zeros] = float("nan")
        return torch.nanmean(sum)
    else:
        return torch.sum(torch.sum(-target * F.log_softmax(logits, -1), -1))


class NpairLoss(nn.Module):
    """N pairs loss"""

    def __init__(self, l2_reg=0.02):
        super(NpairLoss, self).__init__()
        self.l2_reg = l2_reg

    def forward(self, anchor, positive, target, a_indices, p_indices):
        device = torch.device("cuda") if anchor.is_cuda else torch.device("cpu")
        batch_size = anchor.size(0)
        if target is None:
            target = torch.eye(batch_size, dtype=torch.float32).to(device)

        logit = torch.matmul(anchor, torch.transpose(positive, 0, 1))
        loss_ce = cross_entropy(logit, target)

        if a_indices is not None:
            l2_loss = torch.sum(anchor[a_indices] ** 2) / len(a_indices) + torch.sum(
                positive[p_indices] ** 2
            ) / len(p_indices)
        else:
            l2_loss = (torch.sum(anchor**2) + torch.sum(positive**2)) / batch_size

        loss = loss_ce + self.l2_reg * l2_loss * 0.25
        return loss
