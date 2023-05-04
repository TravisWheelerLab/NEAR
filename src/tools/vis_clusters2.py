from tsnecuda import TSNE
import torch 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pdb

embeddings = torch.load('/xdisk/twheeler/daphnedemekas/prefilter/target_embeddings.pt')
embeddings = torch.cat(embeddings[:50],dim=0)

X_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(embeddings)
tsne_df = pd.DataFrame(X_embedded)
# df_subset['tsne-2d-one'] = tsne_results[:,0]
# df_subset['tsne-2d-two'] = tsne_results[:,1]

plt.figure(figsize=(16,10))
sns.scatterplot( x=X_embedded[:,0], y=X_embedded[:,1],palette=sns.color_palette("hls", 10),legend="full",alpha=0.3)
# plt.savefig('tsne2d')