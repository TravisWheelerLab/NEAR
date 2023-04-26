#!/usr/bin/env python3
import logging
import os
import subprocess
import time
from glob import glob
from typing import List

import pandas as pd
import yaml

log = logging.getLogger(__name__)

from argparse import ArgumentParser

import src.utils as utils
from src import array_job_template, name_to_accession_id, single_job_template

DOMTBLOUT_COLS = [0, 4, 11, 19, 20, 22]
DOMTBLOUT_COL_NAMES = [
    "target_name",
    "accession_id",
    "e_value",
    "from",
    "to",
    "description",
]


def emit_and_inject_labels(
    fasta_files: List[str],
    output_directory: str,
    ali_directory: str,
    n: int,
    relent: float = 0.59,
    pid: float = 0.5,
) -> None:
    with open(name_to_accession_id, "r") as src:
        nta = yaml.safe_load(src)
    accession_id_to_name = {v: k for k, v in nta.items()}

    for fasta_file in fasta_files:
        # get neighborhood labels
        labels, _ = utils.fasta_from_file(fasta_file)
        neighborhoods = [utils.parse_labels(labelset)[1:] for labelset in labels]
        neighborhoods_ = []
        for neighborhood_label in neighborhoods:
            if len(neighborhood_label) > 1:
                for nn in neighborhood_label:
                    neighborhoods_.append(nn[0])
        neighborhoods = neighborhoods_
        for neighborhood_label in neighborhoods:
            # convert the PF accession id to a name
            try:
                family_name = accession_id_to_name[neighborhood_label]
                # get correct alignment
                ali_file = os.path.join(
                    ali_directory,
                    f"{family_name}.{pid}-train.sto",
                )
                tmp_hmm_file = random_filename(".")
                # create hmm with correct --ere value (relative entropy)
                # the default is 0.59, per hmmer user guide.

                outf = os.path.join(output_directory, family_name + "_emission.fa")
                if os.path.isfile(outf):
                    pfunc(f"Emission sequences already generated for {family_name}")
                    continue

                if relent != 0.59:
                    subprocess.call(f"hmmbuild --ere {relent} {tmp_hmm_file} {ali_file}".split())
                else:
                    subprocess.call(f"hmmbuild {tmp_hmm_file} {ali_file}".split())

                tmp_emission_path = random_filename(".")
                if os.path.isfile(tmp_hmm_file):

                    cmd = f"hmmemit -o {tmp_emission_path} -N {n} {tmp_hmm_file}"
                    subprocess.call(cmd.split())
                    labels, sequences = utils.fasta_from_file(tmp_emission_path)

                    with open(outf, "w") as dst:
                        for label, sequence in zip(labels, sequences):
                            fasta_header = f">{label} | {neighborhood_label}"
                            fasta_header += "\n" + sequence + "\n"
                            dst.write(fasta_header)

                    pfunc(outf)
                    os.remove(tmp_emission_path)
                    os.remove(tmp_hmm_file)
                else:
                    raise ValueError(f"no hmm file found for {family_name}")

            except KeyError:
                pfunc(f"fasta file {fasta_file} did not contain {family_name}")


def emit_sequences(hmm_file, output_directory, n):
    # save to the same name as the hmm file but with a .fa
    # suffix and in the output directory argument
    output_path = os.path.join(
        output_directory,
        os.path.splitext(os.path.basename(hmm_file))[0] + ".fa",
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
    x = os.path.join(directory, str(time.time()))
    while os.path.isfile(x):
        print("racing..")
        x = os.path.join(directory, str(time.time()))
    return x


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


def parse_domtblout(domtbl):
    """
    Parse a .domtblout file created with hmmsearch -o <tbl>.tblout <seqdb> <hmmdb>
    :param domtbl:
    :type domtbl:
    :return: dataframe containing the rows of the .tblout.
    :rtype: pd.DataFrame
    """

    if os.path.splitext(domtbl)[1] != ".domtblout":
        raise ValueError(f"must pass a .domtblout file, found {domtbl}")

    df = pd.read_csv(
        domtbl,
        skiprows=3,
        header=None,
        delim_whitespace=True,
        usecols=DOMTBLOUT_COLS,
        names=DOMTBLOUT_COL_NAMES,
        skipfooter=10,
        engine="python",
    )

    df = df.dropna()
    # "-" is the empty label
    df.loc[df["description"] != "-", "target_name"] = df["target_name"] + " " + df["description"]

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

    train_hdb_parser = subparsers.add_parser(
        "hdb",
        description="extract training alignment from the alidb and" " create a new hmm",
    )
    train_hdb_parser.add_argument("fasta_file", help="fasta file containing train sequences")
    train_hdb_parser.add_argument("alidb", help="alignment database")
    train_hdb_parser.add_argument("-o", "--overwrite", action="store_true")

    emission_parser = subparsers.add_parser("emit")
    emission_parser.add_argument("hmm_file", help="hmm file to emit sequences from")
    emission_parser.add_argument("output_directory", help="where to save the emitted sequences")
    emission_parser.add_argument("n", type=int, help="number of sequences to emit")

    injection_parser = subparsers.add_parser(
        "inject",
        description="generate neighborhood emission sequences from the neighborhood labels contained in"
        " fasta_files.",
    )
    injection_parser.add_argument("n", help="how many emission sequences to generate")
    injection_parser.add_argument("fasta_files", nargs="+")
    injection_parser.add_argument("output_directory", help="where to save the emitted sequences")
    injection_parser.add_argument("ali_directory", help="where the .sto files are saved")
    injection_parser.add_argument(
        "--relent",
        default=0.59,
        type=float,
        help="relative entropy to use when building hmms",
    )

    return parser


def labels_from_file(fasta_in, fasta_out, domtblout_df):
    """
    Grabs sequences in fasta_in and their corresponding labels in the
    .tblout dataframe (tblout_df), updates the sequence headers with the labels in
     tblout_df, then saves to fasta_out. evalue_threshold controls the precision of the labels.
    :param fasta_in: File containing sequences to label.
    :type fasta_in: str
    :param fasta_out: File to save updated sequences in.
    :type fasta_out: str
    :param domtblout_df: dataframe containing .tblout.
    :type domtblout_df: pd.DataFrame
    :param evalue_threshold: Threshold to keep sequences at.
    :type evalue_threshold: float
    :return: None.
    :rtype: None.
    """
    if os.path.isfile(fasta_out) and os.stat(fasta_out).st_size != 0:
        pfunc(f"Already created labels for {fasta_out}.")
        return

    labels, sequences = utils.fasta_from_file(fasta_in)

    with open(fasta_out, "w") as dst:
        for label, sequence in zip(labels, sequences):
            if " |" in label:
                target_label = label[: label.find(" |")]
            else:
                target_label = label

            assigned_labels = domtblout_df.loc[domtblout_df["target_name"] == target_label]

            assigned_labels = assigned_labels.sort_values(["e_value"])

            if len(assigned_labels) == 0:
                # why are some sequences not classified? They're in Pfam-seed,
                # which means they're manually curated to be part of a family.
                # or the hmmdb can't find them.
                pfunc(f"sequence named {target_label} not found in classification on {fasta_in}")
                target_label = target_label.split()[0]
                # Sometimes the names don't have a description, and aren't found.
                assigned_labels = domtblout_df.loc[domtblout_df["target_name"] == target_label]

                if len(assigned_labels) == 0:
                    # pfunc(
                    #     f"sequence named {target_label} not found in classification on {fasta_in}"
                    # )
                    continue
            # each sequence should have at least one label, but we
            # only need to grab one since one sequence can be associated with
            # multiple pfam accession IDs
            fasta_header = f">{label} |"
            init_len = len(fasta_header)
            prev_evalue = None

            for seq_label, e_value, begin_coord, end_coord in zip(
                assigned_labels["accession_id"],
                assigned_labels["e_value"],
                assigned_labels["from"],
                assigned_labels["to"],
            ):
                if "PF" not in seq_label:
                    raise ValueError(f"Pfam accession ID not found in labels in {domtblout_df}")

                if prev_evalue is None:
                    prev_evalue = float(e_value)
                elif prev_evalue > float(e_value):
                    raise ValueError("Unsorted e-values. Please fix.")

                # removed e-value thresholding (should be done at train time)
                fasta_header += f" {seq_label} ({begin_coord} {end_coord} {e_value})"

            if len(fasta_header) != init_len:
                fasta_header += "\n" + sequence + "\n"
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


def label_with_hmmdb(fasta_file, fasta_outfile, hmmdb, overwrite=False):

    domtblout_path = os.path.splitext(fasta_file)[0] + ".domtblout"
    print(f"domtblout {domtblout_path}")

    if overwrite or not os.path.isfile(domtblout_path):
        print(f"running hmmsearch with {hmmdb}, {fasta_file}, dumping to {domtblout_path}")
        subprocess.call(
            f"hmmsearch -o /dev/null --domtblout {domtblout_path} {hmmdb} {fasta_file}".split()
        )

    domtblout = parse_domtblout(domtblout_path)

    print(f"creating labels. saving to {fasta_outfile}")
    labels_from_file(fasta_file, fasta_outfile, domtblout)


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
        job_id_to_wait_for = self.extract_training_alignments_and_build_hmms(job_id_to_wait_for)
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
                "DEPENDENCY",
                f"#SBATCH --dependency=afterok:{jobid_to_wait_for}",
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
            pfunc(f"Using {self.hmmdb} for labeling instead of concatenating traing hmms.")
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
                "DEPENDENCY",
                f"#SBATCH --dependency=afterok:{jobid_to_wait_for}",
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
        return jobid_to_wait_for

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
                "DEPENDENCY",
                f"#SBATCH --dependency=afterok:{jobid_to_wait_for}",
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
            program_args.hmm_file,
            program_args.output_directory,
            n=program_args.n,
        )
    elif program_args.command == "inject":
        if not os.path.isdir(program_args.output_directory):
            os.makedirs(program_args.output_directory)
        emit_and_inject_labels(
            program_args.fasta_files,
            program_args.output_directory,
            program_args.ali_directory,
            program_args.n,
            relent=program_args.relent,
        )
    else:
        pfunc(program_args)
        program_parser.print_help()
