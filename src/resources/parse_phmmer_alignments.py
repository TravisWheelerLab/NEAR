#!/home/u4/colligan//miniconda3/envs/faiss/bin/python
import logging
import os
from glob import glob

import tqdm

logger = logging.getLogger(__file__)

from Bio import AlignIO

# get hits
# dump to text file formatted as query_name: target hits
tst = "/xdisk/twheeler/colligan/data/prefilter/alignments/*ali"
out_root = "/xdisk/twheeler/colligan/data/prefilter/ungapped_alignments"

query_ungapped = []
target_ungapped = []

already_seen_queries = set()
already_seen_targets = set()

for f in tqdm.tqdm(glob(tst)):
    # this thing is concatenating
    # the different aligned bits
    align = AlignIO.read(f, "stockholm")
    sequences_and_names = []

    q1, q2, t1, t2 = os.path.splitext(os.path.basename(f))[0].split("_")
    query_name = "_".join([q1, q2])
    target_name = "_".join([t1, t2])

    query_gapped_sequence = [s for s in align._records if s.name == query_name]
    target_gapped_sequences = [s for s in align._records if s.name != query_name]

    if len(query_gapped_sequence) > 1:
        raise ValueError(f"something is wrong, for f {f}")
        continue

    if len(query_gapped_sequence) == 0:
        print(f"Skipping {f}. No query sequence.")
        continue

    query_gapped_sequence = query_gapped_sequence[0]
    names_and_sequences = []
    for k, target_sequence in enumerate(target_gapped_sequences):
        keep_idx = []
        for i in range(len(target_sequence)):
            if target_sequence[i] not in ("-", ".") and query_gapped_sequence[i] not in (
                "-",
                ".",
            ):
                keep_idx.append(i)
        # now, add in matches:
        if 256 <= len(keep_idx) <= 300:
            # get the sequences without gaps
            ungapped_ts = "".join([target_sequence[i] for i in keep_idx])
            ungapped_qs = "".join([query_gapped_sequence[i] for i in keep_idx])
            cnt = 0
            qname = query_gapped_sequence.name
            tname = target_sequence.name
            while qname in already_seen_queries:
                qname += f"_{cnt}"
                cnt += 1

            cnt = 0
            while tname in already_seen_targets:
                tname += f"_{cnt}"
                cnt += 1

            already_seen_targets.add(tname)
            already_seen_queries.add(qname)

            query_ungapped.append((qname, ungapped_qs))
            target_ungapped.append((tname, ungapped_ts))

print("Writing queries.")
with open("/xdisk/twheeler/colligan/queries.fa", "w") as dst:
    for name, sequence in query_ungapped:
        dst.write(f">{name}\n{sequence}\n")

print("Writing targets.")
with open("/xdisk/twheeler/colligan/targets.fa", "w") as dst:
    for name, sequence in target_ungapped:
        dst.write(f">{name}\n{sequence}\n")

with open("/xdisk/twheeler/colligan/hits.txt", "w") as dst:
    for (n1, _), (n2, _) in zip(query_ungapped, target_ungapped):
        dst.write(f"{n1} {n2}\n")
