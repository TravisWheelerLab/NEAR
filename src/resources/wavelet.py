import matplotlib.pyplot as plt
import torch
from torchaudio.transforms import MelSpectrogram

from src.utils import amino_char_to_index, fasta_from_file

_, seqs = fasta_from_file(
    "/Users/mac/share/prefilter/src/resources/Q_benchmark2k30k.fa"
)

trans = MelSpectrogram(win_length=32, hop_length=16)

for seq in seqs:
    # what about one-hot encoding?

    encoded = torch.as_tensor([amino_char_to_index[c] for c in seq]).float()
    t1 = trans(encoded)

    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(t1.float())
    ax[1].imshow(torch.log(t1.float()))
    plt.show()
