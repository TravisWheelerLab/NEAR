import logging
import os
import subprocess
from argparse import ArgumentParser
from subprocess import check_output

from tqdm import tqdm

from src.utils import fasta_from_file

logger = logging.getLogger(__name__)

RANDOM_SEED = 42


def hmmbuild(sequence_file, out_path):
    cmd = f"hmmbuild --singlemx --mx BLOSUM62 --seed {RANDOM_SEED} {out_path} {sequence_file}"
    out = check_output(cmd.split())
    logger.debug(f"{out}")


def hmmemit(hmm_file):
    cmd = f"hmmemit --seed {RANDOM_SEED}"
    cmd = cmd.split()
    cmd.append(hmm_file)
    sequence = check_output(cmd).decode("utf-8")
    return sequence


def hmmalign(hmm_file, fasta_file, outfile):
    cmd = f"hmmalign -o {outfile} {hmm_file} {fasta_file}"
    cmd = cmd.split()
    output = check_output(cmd)
    return output


ap = ArgumentParser()
ap.add_argument("file")
args = ap.parse_args()

root = "/xdisk/twheeler/colligan/indel_training_set/"
uniprot_set = f"{root}/split_fasta/{args.file}"
splt = os.path.basename(os.path.splitext(uniprot_set)[0])

names, sequences = fasta_from_file(uniprot_set)
already_seen_names = set()

failcnt = 0
for name, sequence in tqdm(zip(names, sequences), total=len(sequences)):
    if name not in already_seen_names:
        try:
            already_seen_names.add(name)
            sequence_file = f"{root}/sequence_{splt}.fa"
            with open(sequence_file, "w") as dst:
                dst.write(f">{name}\n{sequence}\n")

            hmmfile = f"{root}/hmms/{name.replace(' ', '_').replace('/', '')}_{splt}.hmm"
            hmmbuild(sequence_file, hmmfile)
            emitted_sequence = "".join(hmmemit(hmmfile).split("\n")[1:])
            # dump emitted sequence into file
            sequence_and_emission_file = f"{root}/sequence_and_emission_{splt}.fa"

            with open(sequence_and_emission_file, "w") as dst:
                dst.write(f">{name}\n{sequence}\n")
                dst.write(f">emission\n{emitted_sequence}\n")
            # now align and generate
            alignment_outfile = hmmfile.replace("hmms", "alignments")
            alignment_outfile = alignment_outfile.replace(".hmm", ".sto")
            out = hmmalign(
                hmm_file=hmmfile, fasta_file=sequence_and_emission_file, outfile=alignment_outfile,
            )

        except subprocess.CalledProcessError as e:
            print(f"failure for {name}")
            failcnt += 1

print("fails", failcnt)
