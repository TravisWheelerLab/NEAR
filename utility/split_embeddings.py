import numpy as np
import sys

in_file = sys.argv[1]
out_file = sys.argv[2]
parts = int(sys.argv[3])

embeddings = np.load(in_file)

num_embeddings = len(embeddings)

embeddings_per_file = num_embeddings // parts

for i in range(parts - 1):
	start = embeddings_per_file * i
	end = start + embeddings_per_file
	np.savez(out_file + '_' + str(i) + '.npz', **{str(j - start): embeddings[str(j)] for j in range(start, end)})

start = embeddings_per_file * (parts - 1)
end = num_embeddings
np.savez(out_file + '_' + str(parts - 1) + '.npz', **{str(j - start): embeddings[str(j)] for j in range(start, end)})
