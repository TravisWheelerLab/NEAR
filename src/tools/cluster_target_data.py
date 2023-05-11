import subprocess
import os
import shutil
import tqdm
import argparse
import yaml


def main(trainsequences, evalsequences, clustered_target_dir, target_fastas_dir):
    """Clusters the uniref90 target data into clusters of percent similairty id 0.3

    Requires UCLUST to be installed"""
    t = 0

    for targetfasta in os.listdir(target_fastas_dir):
        targetfastapath = os.path.join(target_fastas_dir, targetfasta)
        clusterpath = f"{clustered_target_dir}/clusters_{t}"
        if not os.path.exists(clusterpath):
            os.mkdir(clusterpath)

        cmd = f"./usearch -cluster_fast {targetfastapath} -id 0.3 -clusters {clusterpath}/cluster_"
        _ = subprocess.run(cmd, shell=True, capture_output=True, check=True,)

        clustered_seqnames = []
        numseqs = 0
        print("Getting sequence names")
        for f in tqdm.tqdm(os.listdir(clusterpath)):
            openf = open(f"{clusterpath}/{f}")
            text = openf.readlines()
            seqnames = []
            for line in text:
                if ">" in line:
                    seqname = text[0].split()[0].strip(">")
                    seqnames.append(seqname)
                    numseqs += 1
            openf.close()
            clustered_seqnames.append(seqnames)

        print("train - test split")

        trainseqs = open(trainsequences, "a")
        valseqs = open(evalsequences, "a")
        print(f"Num seqs: {numseqs}")
        print(f"Num clusters: {len(clustered_seqnames)}")

        i = 0
        j = 0
        for seqlist in clustered_seqnames:
            if i < numseqs * 0.85:
                for seq in seqlist:
                    trainseqs.write(seq + "\n")
                    i += 1
            else:
                for seq in seqlist:
                    valseqs.write(seq + "\n")
                    j += 1
        trainseqs.close()
        print(f"Wrote {i} sequences to train data")

        valseqs.close()

        print(f"Wrote {j} sequences to eval data")

        shutil.rmtree(clusterpath)

        t += 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--target_fasta")
    parser.add_argument("--cluster_savedir")
    parser.add_argument("--train_savepath")
    parser.add_argument("--eval_savepath")

    args = parser.parse_args()

    if not os.path.exists(args.train_savepath):
        trainseqs = open(args.train_savepath, "w")
        trainseqs.close()
    if not os.path.exists(args.eval_savepath):
        valseqs = open(args.eval_savepath, "w")
        valseqs.close()

    if not os.path.exists(args.cluster_savedir):
        print(f"Making target directory {args.cluster_savedir}")
        os.mkdir(args.cluster_savedir)

    assert os.path.exists(
        "./usearch"
    ), "Usearch needs to be installed to your home directory \
        in order to run this script. Please install Usearch or otherwise change the command to point \
        to where Usearch is located on your machine. For more information see README."

    main(args.train_savepath, args.eval_savepath, args.cluster_savedir, args.target_fasta)
