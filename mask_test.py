import torch

# emulate the masking process

# features:
# batch_size x embed_dim x L
features = torch.cat([torch.arange(5).unsqueeze(1)] * 3, dim=1)
feats = []
for _ in range(4):
    feats.append(features.unsqueeze(0).clone())

features = torch.cat(feats, dim=0)
print(features.shape)
print(features)
exit()

masks = []

for i in range(1, 11):
    x = torch.ones(10)
    x[:i] = 0
    masks.append(x.unsqueeze(0))

mask = torch.cat(masks, dim=0).bool()
print(mask)
features = features[~mask].reshape(-1, 10)
print(features)
