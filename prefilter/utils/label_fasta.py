#!/usr/bin/env python3
import pdb
import subprocess
import pandas as pd
import logging
import os
import time
from glob import glob

log = logging.getLogger(__name__)

from argparse import ArgumentParser
from prefilter import array_job_template, single_job_template
import prefilter.utils as utils


def emit_sequences(hmm_file, output_directory, n):
    # save to the same name as the hmm file but with a .fa
    # suffix and in the output directory argument
    output_path = os.path.join(
        output_directory, os.path.splitext(os.path.basename(hmm_file))[0] + ".fa"
    )
    cmd = f"hmmemit -o {output_path} -N {n} {hmm_file}"
    subprocess.call(cmd.split())


def random_filename(directory="/tmp/"):
    """
    Generate a random filename for temporary use.

    :param directory: Which directory to place the file in
    :type directory: str
    :return: random filename in directory
    :rtype: str
    """
    return os.path.join(directory, str(time.time()))


def job_completed(slurm_jobid):
    """
    Returns whether or not a slurm job has completed.
    :param slurm_jobid: id of job you want to check.
    :type slurm_jobid: int.
    :return: whether or not the job has completed.
    :rtype: bool.
    """
    slurm_jobid = int(slurm_jobid)
    job_status = subprocess.check_output(
        f"sacct --format State -u {os.environ['USER']} -j {slurm_jobid}".split()
    )
    job_status = job_status.decode("utf-8")
    # TODO: Figure out if this is going to always work for slurm jobs.
    return "COMPLETED" in job_status


def pfunc(str):
    print(str)


TBLOUT_COL_NAMES = [
    "target_name",
    "query_name",
    "accession_id",
    "e_value",
    "description",
]
TBLOUT_COLS = [0, 2, 3, 4, 18]


def parse_tblout(tbl):
    """
    Parse a .tblout file created with hmmsearch -o <tbl>.tblout <seqdb> <hmmdb>
    :param tbl: .tblout filename.
    :type tbl: str
    :return: dataframe containing the rows of the .tblout.
    :rtype: pd.DataFrame
    """
    if os.path.splitext(tbl)[1] != ".tblout":
        raise ValueError(f"must pass a .tblout file, found {tbl}")

    df = pd.read_csv(
        tbl,
        skiprows=3,
        header=None,
        delim_whitespace=True,
        usecols=TBLOUT_COLS,
        names=TBLOUT_COL_NAMES,
    )
    # - is empty label
    df["target_name"].loc[df["description"] != "-"] = (
        df["target_name"] + " " + df["description"]
    )

    return df


def create_parser():
    """
    Creates the argument parser.
    :return: Argument parser
    :rtype: argparse.ArgumentParser
    """
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(title="action", dest="command")

    pipeline_parser = subparsers.add_parser(name="generate")

    pipeline_parser.add_argument("-a", "--afa_directory")
    pipeline_parser.add_argument("-c", "--clustered_output_directory")
    pipeline_parser.add_argument("-t", "--training_data_output_directory")
    pipeline_parser.add_argument("-p", "--pid")
    pipeline_parser.add_argument(
        "-adb", "--alidb", help="database of alignments in stockholm format"
    )
    pipeline_parser.add_argument("-db", "--hmmdb", default=None)
    pipeline_parser.add_argument("-e", "--evalue_threshold", type=float, default=1e-5)

    split_parser = subparsers.add_parser(name="split", add_help=False)
    split_parser.add_argument("-a", "--aligned_fasta_file", type=str)
    split_parser.add_argument(
        "-p",
        "--pid",
        type=float,
        help="percent identity to split the aligned fasta file",
        required=True,
    )
    split_parser.add_argument(
        "-c",
        "--clustered_output_directory",
        type=str,
        help="where to save the clustered .fa files.",
        required=True,
    )

    label_parser = subparsers.add_parser("label")

    label_parser.add_argument("fasta_file", help="fasta file to label")
    label_parser.add_argument(
        "hmmdb",
        default=None,
        help="hmm database to search the aligned fasta file against",
    )
    label_parser.add_argument(
        "-o",
        "--fasta_output_directory",
        help="where to save the labeled fasta files",
        required=True,
    )

    train_hdb_parser = subparsers.add_parser("hdb")
    train_hdb_parser.add_argument(
        "fasta_file", help="fasta file containing train sequences"
    )
    train_hdb_parser.add_argument("alidb", help="alignment database")
    train_hdb_parser.add_argument("-o", "--overwrite", action="store_true")

    emission_parser = subparsers.add_parser("emit")
    emission_parser.add_argument("hmm_file", help="hmm file to emit sequences from")
    emission_parser.add_argument(
        "output_directory", help="where to save the emitted sequences"
    )
    emission_parser.add_argument("n", type=int, help="number of sequences to emit")
    return parser


def labels_from_file(fasta_in, fasta_out, tblout_df, evalue_threshold=1e-5):
    """
    Grabs sequences in fasta_in and their corresponding labels in the
    .tblout dataframe (tblout_df), updates the sequence headers with the labels in
     tblout_df, then saves to fasta_out. evalue_threshold controls the precision of the labels.
    :param fasta_in: File containing sequences to label.
    :type fasta_in: str
    :param fasta_out: File to save updated sequences in.
    :type fasta_out: str
    :param tblout_df: dataframe containing .tblout.
    :type tblout_df: pd.DataFrame
    :param evalue_threshold: Threshold to keep sequences at.
    :type evalue_threshold: float
    :return: None.
    :rtype: None.
    """
    if os.path.isfile(fasta_out):
        pfunc(f"Already created labels for {fasta_out}.")
        return

    labels, sequences = utils.fasta_from_file(fasta_in)

    with open(fasta_out, "w") as dst:
        for label, sequence in zip(labels, sequences):
            if " |" in label:
                target_label = label[: label.find(" |")]
            else:
                target_label = label

            assigned_labels = tblout_df.loc[tblout_df["target_name"] == target_label]

            if len(assigned_labels) == 0:
                # why are some sequences not classified? They're in Pfam-seed,
                # which means they're manually curated to be part of a family.
                # or the hmmdb can't find them.
                pfunc(
                    f"sequence named {target_label} not found in classification on {fasta_in}"
                )
                target_label = target_label.split()[0]
                # Sometimes the names don't have a description, and aren't found.
                assigned_labels = tblout_df.loc[
                    tblout_df["target_name"] == target_label
                ]
                if len(assigned_labels) == 0:
                    pfunc(
                        f"sequence named {target_label} not found in classification on {fasta_in}"
                    )
                continue
            # each sequence should have at least one label, but we
            # only need to grab one since one sequence can be associated with
            # multiple pfam accession IDs
            fasta_header = f">{label} | "

            labelset = []

            for seq_label, e_value in zip(
                assigned_labels["accession_id"], assigned_labels["e_value"]
            ):
                if "PF" not in seq_label:
                    raise ValueError(
                        f"Pfam accession ID not found in labels in {tblout_df}"
                    )

                if float(e_value) <= evalue_threshold:
                    labelset.append(seq_label)

            if len(labelset):
                fasta_header += " ".join(labelset) + "\n" + sequence + "\n"
                dst.write(fasta_header)


def cluster_and_split_sequences(aligned_fasta_file, clustered_output_directory, pid):
    """
    Use carbs (https://github.com/TravisWheelerLab/carbs) to split the sequences in the aligned fasta file at
    percent identity pid.
    :param aligned_fasta_file:
    :type aligned_fasta_file:
    :param clustered_output_directory:
    :type clustered_output_directory:
    :param pid:
    :type pid:
    :return:
    :rtype:
    """
    output_template = (
        os.path.join(
            clustered_output_directory,
            os.path.splitext(os.path.basename(aligned_fasta_file))[0],
        )
        + ".{}-{}.fa"
    )

    os.makedirs(clustered_output_directory, exist_ok=True)

    ddgm_path = os.path.splitext(aligned_fasta_file)[0] + ".ddgm"
    # cluster the .afa if we can't find the .ddgm
    if not os.path.isfile(ddgm_path) or os.stat(ddgm_path).st_size == 0:
        subprocess.call(f"carbs cluster {aligned_fasta_file}".split())

    if not os.path.isfile(output_template.format(pid, "train")):
        cmd = f"carbs split -T argument --split_test --output_path {clustered_output_directory} {aligned_fasta_file} {pid}"
        subprocess.call(cmd.split())
    else:
        pfunc(f"already created {output_template.format(pid, 'train')}")


def label_with_hmmdb(fasta_file, fasta_outfile, hmmdb):

    tblout_path = os.path.splitext(fasta_file)[0] + ".tblout"

    if not os.path.isfile(tblout_path) or os.stat(tblout_path).st_size == 0:
        subprocess.call(
            f"hmmsearch -o /dev/null --tblout {tblout_path} {hmmdb} {fasta_file}".split()
        )

    tblout = parse_tblout(tblout_path)

    labels_from_file(fasta_file, fasta_outfile, tblout)


def extract_ali_and_create_hmm(fasta_file, alidb, overwrite=False):

    if "train" not in os.path.basename(fasta_file):
        raise ValueError(f'{fasta_file} does not have "train" in it.')

    stockholm_out = os.path.splitext(fasta_file)[0] + ".sto"

    if not os.path.isfile(stockholm_out) or overwrite:
        # extract the alignment from the alidb that has the same
        # name as the fasta file
        family_name = os.path.basename(fasta_file)
        family_name = family_name[: family_name.find(".")]
        train_alignment_temp_file = random_filename()
        cmd = f"esl-afetch -o {train_alignment_temp_file} {alidb} {family_name}"
        assert subprocess.call(cmd.split()) == 0

        train_name_file = random_filename()
        train_names, _ = utils.fasta_from_file(fasta_file)

        with open(train_name_file, "w") as dst:
            for name in train_names:
                dst.write(f"{name.split()[0]}\n")

        pfunc(len(train_names))
        if len(train_names):
            cmd = f"esl-alimanip -o {stockholm_out} --seq-k {train_name_file} {train_alignment_temp_file}"
            # check for successful exit code
            assert subprocess.call(cmd.split()) == 0
        os.remove(train_name_file)
    else:
        pfunc(f"{stockholm_out} already exists.")

    hmm_out_path = os.path.splitext(fasta_file)[0] + ".hmm"

    if not os.path.isfile(hmm_out_path) or overwrite:
        # create the hmm
        cmd = f"hmmbuild -o /dev/null -n 1 {hmm_out_path} {train_alignment_temp_file}"
        assert subprocess.call(cmd.split()) == 0
        os.remove(train_alignment_temp_file)
    else:
        pfunc(f"{hmm_out_path} already exists.")


class Generator:
    def __init__(
        self,
        aligned_fasta_directory,
        clustered_output_directory,
        training_data_output_directory,
        pid,
        alidb,
        evalue_threshold,
        hmmdb=None,
        poll_interval=1,
    ):

        self.aligned_fasta_directory = aligned_fasta_directory
        self.clustered_output_directory = clustered_output_directory
        self.training_data_output_directory = training_data_output_directory
        self.pid = pid
        self.alidb = alidb
        self.hmmdb = hmmdb
        self.evalue_threshold = evalue_threshold
        self.poll_interval = poll_interval
        self.jobid_to_wait_for = None

        self.files_to_delete = []

        # split and cluster these afas using a slurm array job # requiring a slurm template.
        job_id_to_wait_for = self.split_afa_array_job()
        job_id_to_wait_for = self.extract_training_alignments_and_build_hmms(
            job_id_to_wait_for
        )
        job_id_to_wait_for = self.concatenate_hmms(job_id_to_wait_for)
        job_id_to_wait_for = self.label_with_hmm(job_id_to_wait_for)
        self.delete(job_id_to_wait_for)

    def _dump_data(self, filename, data):
        if isinstance(data, list):
            with open(filename, "w") as dst:
                dst.write("\n".join(data))
        elif isinstance(data, str):
            with open(filename, "w") as dst:
                dst.write(data)
        else:
            raise ValueError("only accepts strings and lists")

    def split_afa_array_job(self):
        random_f = self._random_file(".")
        # use ls to dump all the .afas for splitting into a name file
        afa_files = glob(os.path.join(self.aligned_fasta_directory, "*.afa"))
        self._dump_data(random_f, afa_files)

        slurm_script = array_job_template.replace("ARRAY_JOBS", str(len(afa_files)))
        slurm_script = slurm_script.replace("ARRAY_INPUT_FILE", random_f)
        run_cmd = (
            f"/home/tc229954/anaconda/envs/prefilter/bin/python "
            f"/home/tc229954/share/prefilter/prefilter/utils/label_fasta.py split "
            f"-a $f -p {self.pid} -c {self.clustered_output_directory}"
        )
        slurm_script = slurm_script.replace("RUN_CMD", run_cmd)
        slurm_script = slurm_script.replace("DEPENDENCY", "")
        slurm_script = slurm_script.replace("ERR", "split_afa.err")

        # write slurm script to file
        slurm_file = self._random_file(".")

        self._dump_data(slurm_file, slurm_script)

        pfunc("Submitting clustering array job script.")
        return self._submit(slurm_file)

    def extract_training_alignments_and_build_hmms(self, jobid_to_wait_for):

        train_names = glob(os.path.join(self.clustered_output_directory, "*train.fa"))
        random_train_fasta_file = self._random_file(".")

        self._dump_data(random_train_fasta_file, train_names)

        slurm_script = array_job_template.replace("ARRAY_JOBS", str(len(train_names)))
        slurm_script = slurm_script.replace("ARRAY_INPUT_FILE", random_train_fasta_file)

        # slurm script to build the hmms
        run_cmd = (
            f"/home/tc229954/anaconda/envs/prefilter/bin/python "
            f"/home/tc229954/share/prefilter/prefilter/utils/label_fasta.py hdb "
            f"$f {self.alidb}"
        )

        slurm_script = slurm_script.replace("RUN_CMD", run_cmd)
        slurm_script = slurm_script.replace("ERR", "build_hmm.err")

        if jobid_to_wait_for is not None:
            slurm_script = slurm_script.replace(
                "DEPENDENCY", f"#SBATCH --dependency=afterok:{jobid_to_wait_for}"
            )
        else:
            slurm_script = slurm_script.replace("DEPENDENCY", "")
        slurm_file = self._random_file(".")
        self._dump_data(slurm_file, slurm_script)

        pfunc("Submitting hmm building array job script.")
        return self._submit(slurm_file)

    def concatenate_hmms(self, jobid_to_wait_for):
        jobid = None
        if self.hmmdb is None:
            output_hmm_file = f"{self.clustered_output_directory}/Pfam-{self.pid}.hmm"
            if os.path.isfile(output_hmm_file):
                pfunc(
                    f"Found concatenation of hmms at {output_hmm_file}. Continuing on to next step without recreating."
                )
            else:
                run_cmd = f"for f in {self.clustered_output_directory}/*.hmm; do cat $f >> {output_hmm_file}; done"
                bash_script = single_job_template.replace("RUN_CMD", run_cmd)
                if jobid_to_wait_for is not None:
                    bash_script = bash_script.replace(
                        "DEPENDENCY",
                        f"#SBATCH --dependency=afterok:{jobid_to_wait_for}",
                    )
                else:
                    bash_script = bash_script.replace("DEPENDENCY", "")
                bash_random_file = self._random_file(".")
                self._dump_data(bash_random_file, bash_script)
                jobid = self._submit(bash_random_file)

            self.hmmdb = output_hmm_file
        else:
            pfunc(
                f"Using {self.hmmdb} for labeling instead of concatenating traing hmms."
            )
        return jobid

    def label_with_hmm(self, jobid_to_wait_for):
        # use the hmmdb that was created by concatenate_hmms or passed
        # .afa files to label:
        fa_files = glob(os.path.join(self.clustered_output_directory, "*.fa"))

        random_f = self._random_file(".")
        self._dump_data(random_f, fa_files)

        slurm_script = array_job_template.replace("ARRAY_JOBS", str(len(fa_files)))
        slurm_script = slurm_script.replace("ARRAY_INPUT_FILE", random_f)
        run_cmd = (
            f"/home/tc229954/anaconda/envs/prefilter/bin/python "
            f"/home/tc229954/share/prefilter/prefilter/utils/label_fasta.py label "
            f"$f {self.hmmdb} -o {self.training_data_output_directory}"
        )
        slurm_script = slurm_script.replace("RUN_CMD", run_cmd)
        slurm_script = slurm_script.replace("ERR", "label.err")

        if jobid_to_wait_for is not None:
            slurm_script = slurm_script.replace(
                "DEPENDENCY", f"#SBATCH --dependency=afterok:{jobid_to_wait_for}"
            )
        else:
            slurm_script = slurm_script.replace("DEPENDENCY", "")

        # write slurm script to file
        slurm_file = self._random_file(".")

        self._dump_data(slurm_file, slurm_script)

        pfunc(
            f"Submitting labeling array job script. Labeling each .fa in {self.clustered_output_directory} with {self.hmmdb}"
        )
        jobid_to_wait_for = self._submit(slurm_file)
        return self.jobid_to_wait_for

    def _submit(self, slurm_script):
        slurm_jobid = subprocess.check_output(
            f"sbatch --wait --parsable {slurm_script}",
            shell=True,
        )
        return int(slurm_jobid)

    def _random_file(self, directory="/tmp/"):
        f = random_filename(directory)
        self.files_to_delete.append(f)
        return f

    def delete(self, jobid_to_wait_for):
        random_f = self._random_file(".")
        self._dump_data(random_f, self.files_to_delete)
        run_cmd = f"cat {random_f} | while read line; do rm $line; done"
        bash_script = single_job_template.replace("RUN_CMD", run_cmd)

        if jobid_to_wait_for is not None:
            bash_script = bash_script.replace(
                "DEPENDENCY", f"#SBATCH --dependency=afterok:{jobid_to_wait_for}"
            )
        else:
            bash_script = bash_script.replace("DEPENDENCY", "")

        bash_random_f = self._random_file(".")
        self._dump_data(bash_random_f, bash_script)
        self._submit(bash_random_f)
        os.remove(bash_random_f)


if __name__ == "__main__":
    program_parser = create_parser()
    program_args = program_parser.parse_args()
    if program_args.command == "split":
        cluster_and_split_sequences(
            program_args.aligned_fasta_file,
            program_args.clustered_output_directory,
            program_args.pid,
        )
    elif program_args.command == "label":
        fasta_outf = os.path.join(
            program_args.fasta_output_directory,
            os.path.basename(program_args.fasta_file),
        )
        os.makedirs(program_args.fasta_output_directory, exist_ok=True)
        label_with_hmmdb(program_args.fasta_file, fasta_outf, program_args.hmmdb)
    elif program_args.command == "hdb":
        extract_ali_and_create_hmm(
            program_args.fasta_file, program_args.alidb, program_args.overwrite
        )
    elif program_args.command == "generate":
        Generator(
            aligned_fasta_directory=program_args.afa_directory,
            clustered_output_directory=program_args.clustered_output_directory,
            training_data_output_directory=program_args.training_data_output_directory,
            pid=program_args.pid,
            alidb=program_args.alidb,
            evalue_threshold=program_args.evalue_threshold,
            hmmdb=program_args.hmmdb,
        )
    elif program_args.command == "emit":
        if not os.path.isdir(program_args.output_directory):
            os.makedirs(program_args.output_directory)
        emit_sequences(
            program_args.hmm_file, program_args.output_directory, n=program_args.n
        )
    else:
        pfunc(program_args)
        program_parser.print_help()
