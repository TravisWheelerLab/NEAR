import torch
import os
import matplotlib.pyplot as plt
import re
from glob import glob


def read_posterior_file(file_path):
    matrix = []
    with open(file_path, "r") as file:
        found_dump = False
        read_line_count = 0
        for line in file:
            if not found_dump:
                if "POSTERIOR DUMP" in line:
                    found_dump = True
                    continue
            else:
                row = []
                if "END DUMP" in line:
                    break
                if read_line_count >= 2:
                    line = line.strip()
                    line = re.split(" +", line)
                    if len(line) == 1:
                        continue
                    if line[1] != "M":
                        continue
                    for i in range(2, len(line)):
                        if line[i] == "-inf":
                            val = -100
                        else:
                            val = float(line[i])

                        row.append(val)

                    matrix.append(row)
                read_line_count += 1
    return torch.tensor(matrix)[1:, 1:]


for file in glob("/Users/mac/Dropbox/notebook/prefilter/2022-05-17/transformer/*txt"):
    print(file)
    posts = read_posterior_file(file)
    plt.imshow(torch.exp(posts))
    plt.title(f"posterior probs, input_file: {os.path.basename(file)}")
    plt.colorbar()
    plt.savefig(f"{os.path.splitext(file)[0]}.png")
    plt.close()
