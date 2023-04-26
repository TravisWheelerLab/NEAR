"""Tools to parse stdout of hmmer to generate alignments data"""
import argparse
import logging
import tqdm
from Bio import SearchIO
from src.data.hmmerhits import FastaFile

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)


def parse_stdout(
    stdout_path: str,
    alignment_file_dir: str,
    querysequences: dict = None,
    targetsequences: dict = None,
    write_full_seq=True,
):
    """Parse the saved off stdout files from running hmmer

    alignment_file_dir: the directory of the txt file where to save the alignment.
        the alignments will be saved as query_id-target_id

    stdout_path: the path of the hmmer stdout txt file that was saved off from running phmmer

    write_full_seq: This is True if you want to save the original full length sequence along with the alignment

    querysequences, targetsequences: dictionaries of type {id: sequence}. This is needed if write_full_seq = True

    The output files saved in alignment_file_dir will be of structure:

    >query_id & target_id
    aligned_seq1
    aligned_seq2
    full_seq1
    full_seq2
    """

    if write_full_seq:
        assert (
            querysequences is not None
        ), "No querysequences dictionary is given. This is needed to write full sequences"
        assert (
            targetsequences is not None
        ), "No targetsequences dictionary is given. This is needed to write full sequences"

    result = SearchIO.parse(stdout_path, "hmmer3-text")

    for qresult in tqdm.tqdm(result):
        query_id = qresult.id
        for hit in qresult:
            target_id = hit.id

            if hit.evalue > 1:
                continue

            alignment_file_path = f"{alignment_file_dir}/{query_id}-{target_id}.txt"

            with open(alignment_file_path, "w") as alignment_file:
                for hsp in hit:
                    alignments = hsp.aln
                    seq1 = str(alignments[0].seq)
                    seq2 = str(alignments[1].seq)
                    alignment_file.write(">" + query_id + " & " + target_id + "\n")
                    alignment_file.write(seq1 + "\n")
                    alignment_file.write(seq2 + "\n")
                    if write_full_seq:
                        fullseq1 = querysequences[query_id]
                        fullseq2 = targetsequences[target_id]
                        alignment_file.write(fullseq1 + "\n")
                        alignment_file.write(fullseq2 + "\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("stdout_path")
    parser.add_argument("save_dir")
    parser.add_argument("query_fasta")
    parser.add_argument("target_fasta")
    args = parser.parse_args()

    queryfasta = FastaFile(args.query_fasta)
    querysequences = queryfasta.data
    targetfasta = FastaFile(args.target_fasta)
    targetsequences = targetfasta.data

    args = parser.parse_args()

    parse_stdout(args.stdout_path, args.save_dir, querysequences, targetsequences)
