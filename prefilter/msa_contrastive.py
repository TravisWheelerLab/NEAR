import torch
from sys import stdout
import numpy as np
import os
from glob import glob
from sklearn.neighbors import BallTree
import yaml

from prefilter.models import ResNet1d, SupConLoss
from prefilter.utils import ContrastiveGenerator, pad_contrastive_batches

logo_path = "/home/tc229954/data/prefilter/pfam/seed/clustered/0.5/"
fasta_path = "/home/tc229954/data/prefilter/pfam/seed/model_comparison/training_data_no_evalue_threshold/200_file_subset"

fasta_files = glob(os.path.join(fasta_path, "*-train.fa"))
logo_files = glob(os.path.join(logo_path, "*.logo"))

dataset = ContrastiveGenerator(fasta_files, logo_files, name_to_class_code=None)

batch_size = 256
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_contrastive_batches
)

device = "cuda"
encoder = ResNet1d().to(device)
optim = torch.optim.SGD(encoder.parameters(), lr=0.5, momentum=0.9)
i = 0
cos_sim = torch.nn.CosineSimilarity(dim=-1)
crit = SupConLoss()
# i think this is correct...

for epoch in range(100):
    mn_loss = 0
    j = 0
    for features, labels in dataloader:
        j += 1
        optim.zero_grad()
        if features.shape[0] != 2 * batch_size:
            continue

        # 2*batch_sizex256
        embeddings = encoder(features.to(device).float())
        # normalize over the embedding dimension

        embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        # then stack into pos/pos pairs
        embeddings = embeddings.view(batch_size, 2, 128)

        loss = crit(embeddings, labels)
        mn_loss += loss.item()
        loss.backward()
        optim.step()

    print(f"epoch: {epoch}, loss: {loss.item()}, mn_loss: {mn_loss/j}")

# eval. all logos
logos = [dataset.name_to_logo[n] for n in dataset.names]
names = dataset.names
logo_embeddings = np.zeros((len(logos), 128))
with torch.no_grad():
    for i, logo in enumerate(logos):
        embedding = encoder(torch.as_tensor(logo).to(device).float().unsqueeze(0))
        logo_embeddings[i] = embedding.squeeze().detach().cpu().numpy()

# ball tree doesn't natively support cosine similarity as a metric,
# so just do the matmul for now.
logo_embeddings = torch.as_tensor(logo_embeddings).to(device).float()
logo_embeddings = torch.nn.functional.normalize(logo_embeddings, dim=-1)

total = 0
correct = 0
with torch.no_grad():
    for i, name in enumerate(names):
        sequences = dataset.name_to_sequences[name]
        total += len(sequences)
        for sequence in sequences:
            predicted_embedding = encoder(
                torch.as_tensor(sequence).to(device).float().unsqueeze(0)
            )
            predicted_embedding = predicted_embedding / torch.norm(
                predicted_embedding, dim=-1
            )
            nearest_neighbors = torch.matmul(
                logo_embeddings, predicted_embedding.T
            ).squeeze()
            nearest_neighbors_idx = torch.argsort(nearest_neighbors).cpu().numpy()
            if i in set(nearest_neighbors_idx[-5:]):
                correct += 1

print(correct, total, correct / total)
