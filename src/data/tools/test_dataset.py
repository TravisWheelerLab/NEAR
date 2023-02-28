from src.datasets.alignmentgenerator import AlignmentGeneratorWithIndels
import torch
import pdb
from src.utils import pluginloader
from src import models
import os
HOME = os.environ["HOME"]

train_dataset = AlignmentGeneratorWithIndels(ali_path = "/xdisk/twheeler/daphnedemekas/train_paths.txt",seq_len = 128)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    collate_fn=train_dataset.collate_fn(),
    batch_size = 32,
    num_workers = 6,
    drop_last = True
)
dataiter = iter(train_dataloader)

seq1, seq1_indices, seq2, seq2_indices = next(dataiter)


model_dict = {
    m.__name__: m for m in pluginloader.load_plugin_classes(models, pl.LightningModule)
}

model_class = model_dict["ResNet1d"]
checkpoint_path = f"{HOME}/prefilter/ResNet1d/4/checkpoints/best_loss_model.ckpt"
device = "cuda"

model = model_class(learning_rate = 1e-5,log_interval = 100,in_channels = 20)
print("Loaded model")
features = torch.cat([seq1, seq2], dim = 0)

embedding = model(features) 

feature1_indices = seq1_indices
feature2_indices = seq2_indices

seq_len = feature1_indices.shape[1]

for batch_idx in range(feature1_indices.shape[0]):
    feature1_indices[batch_idx] += batch_idx * seq_len
    feature2_indices[batch_idx] += batch_idx * seq_len
#feature1_indices = feature1_indices.contiguous().view(-1,1)
#feature2_indices = feature2_indices.contiguous().view(-1,1)
#mask = torch.eq(feature1_indices, feature2_indices.T)
l1 = torch.cat(torch.unbind(feature1_indices, dim=0))
l2 = torch.cat(torch.unbind(feature2_indices, dim=0))
labelmat = torch.eq(l1.unsqueeze(1), l2.unsqueeze(0)).float()

batch_size = features.shape[0]

mask = labelmat
contrast_count = features.shape[1]
contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

anchor_feature = contrast_feature
anchor_count = contrast_count

# compute logits
anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), 0.07)
# for numerical stability
logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
logits = anchor_dot_contrast - logits_max.detach()

# tile mask
mask = mask.repeat(anchor_count, contrast_count)
# mask-out self-contrast cases
logits_mask = torch.scatter(torch.ones_like(mask),1,torch.arange(batch_size * anchor_count).view(-1, 1).to(device),0,)
mask = mask * logits_mask

# compute log_prob
exp_logits = torch.exp(logits) * logits_mask
log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

# compute mean of log-likelihood over positive
mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
# loss
loss = -(0.07 / 0.07) * mean_log_prob_pos

loss = loss.view(anchor_count, batch_size).nanmean()

pdb.set_trace()








feature1_indices = torch.stack(seq1_indices).T
feature2_indices = torch.stack(seq2_indices).T

seq_len = feature1_indices.shape[1]

for batch_idx in range(feature1_indices.shape[0]):
    feature1_indices[batch_idx] += batch_idx * seq_len
    feature2_indices[batch_idx] += batch_idx * seq_len

feature1_indices = feature1_indices.contiguous().view(-1,1)
feature2_indices = feature2_indices.contiguous().view(-1,1)
mask = torch.eq(feature1_indices, feature2_indices.T)
features = torch.cat([seq1, seq2], dim = 0)
mask = mask.repeat(2, 2)
pdb.set_trace()
