#%%import torch 
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
import seaborn as sns

embeddings = torch.load('/xdisk/twheeler/daphnedemekas/prefilter/target_embeddings.pt')

def sequence_representation(embeddings, max_len = 100):


    embed = [e[:max_len] for e in embeddings[::50] if len(e) >= max_len ]
    embed_t = torch.stack(embed,dim=0)
    embed_tt = embed_t.reshape(embed_t.shape[0], embed_t.shape[1] * embed_t.shape[2])
    return embed_tt

def amino_representation(embeddings):
    amino_embeddings = torch.cat(embeddings[::20000], dim = 0)
    return amino_embeddings

seq_embeddings = amino_representation(embeddings)

tsne = TSNE(3, verbose=1)
tsne_proj = tsne.fit_transform(seq_embeddings)

print(f"Projection shape : {tsne_proj.shape})")
sns.set_style ("darkgrid")
plt.figure(figsize = (5, 4))
plot_axes = plt.axes(projection = '3d')
plot_axes.scatter3D(tsne_proj[:,0],tsne_proj[:,1], tsne_proj[:,2], alpha = 0.3)
plot_axes.set_xlabel('x')
plot_axes.set_ylabel('y')
plot_axes.set_zlabel('z')
plt.xlim(-100,0)
plt.ylim(0,400)
plot_axes.set_zlim(-100,0)
plt.title("Amino-wise t-SNE ")

plt.savefig('tnse3d_.png')
