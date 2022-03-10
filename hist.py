from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import prefilter
import prefilter.utils as utils

fasta_files = glob("/home/tc229954/max_hmmsearch/200_file_subset/*train*")

e_values = []
i = 0
for fasta_file in fasta_files:
    i += 1
    labels, sequences = utils.fasta_from_file(fasta_file)
    for labelstring, sequence in zip(labels, sequences):
        labelset = utils.parse_labels(labelstring)

        for item in labelset:
            x = float(item[-1])
            if x == 0:
                print(x)
            else:
                e_values.append(x)

fig, ax = plt.subplots(figsize=(12, 10))

ax.semilogy()
ax.hist(np.log10(e_values), bins=100, histtype='step')

ax.set_xlabel("log$_{10}$(e_value)")
ax.set_ylabel("log(count)")
plt.savefig("train_evalue_hist.png", bbox_inches="tight")
plt.close()

