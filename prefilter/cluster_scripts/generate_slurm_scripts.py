import os
from glob import glob
from argparse import ArgumentParser


SLURM_DIRECTIVES = "#!/bin/bash\n#SBATCH --job-name={}\n#SBATCH --output={}.out\n#SBATCH --partition=wheeler_lab_small_cpu,wheeler_lab_large_cpu\n"


def create_parser():
    ap = ArgumentParser(prog="generate data for one family.")
    ap.add_argument("aligned_fasta", help=".afa file to split with `carbs split`.")
    ap.add_argument(
        "pid", help="what percent identity to split the .afa file at.", type=float
    )
    ap.add_argument(
        "-o", "--output_directory", help="where to save the labeled fasta files."
    )
    ap.add_argument(
        "-c", "--carbs_output_directory", help="where to save the split fasta files."
    )
    ap.add_argument(
        "-s",
        "--slurm_output_directory",
        help="where to save the created slurm file (saved, by default, as the "
        "basename of the aligned fasta + .sh",
    )
    ap.add_argument(
        "-d",
        "--domtblout_path",
        help="directory with the domtblouts (if they aren't already in the same place as the .afas)",
    )
    ap.add_argument(
        "--from_glob",
        action="store_true",
        help="generate all slurm scripts from all the .afa files"
        " in the same directory as the aligned fasta",
    )
    return ap


def create_slurm_script(
    afa, pid, carbs_output_directory, fasta_output_directory, slurm_out_directory, domtblout_path

):
    """
    Write a single slurm script containing details of data generation.
    Assumes that a .domtblout file with basename(afa) + .domtblout resides in the same directory as the .afa file.
    Does not run hmmer to produce the domtblout.
    :param afa: .afa file to cluster, split, and then label.
    :type afa: str
    :param pid: percent identity at which to split the .afa
    :type pid: float
    :param carbs_output_directory: where to save the raw clustered data.
    :type carbs_output_directory: str
    :param fasta_output_directory: where to save the labeled fasta files
    :type fasta_output_directory: str
    :param slurm_out_directory: where to save the generated slurm script.
    :type slurm_out_directory: str
    :return: None
    :rtype: None
    """

    job_name = os.path.splitext(os.path.basename(afa))[0]

    os.makedirs(carbs_output_directory, exist_ok=True)
    os.makedirs(slurm_out_directory, exist_ok=True)
    os.makedirs(fasta_output_directory, exist_ok=True)

    cmd_list = [
        SLURM_DIRECTIVES.format(job_name, os.path.join(slurm_out_directory, job_name))
    ]

    slurm_outfile = os.path.join(
        slurm_out_directory, os.path.splitext(os.path.basename(afa))[0] + ".sh"
    )

    if not os.path.isfile(os.path.splitext(afa)[0] + ".ddgm"):
        # cluster the data.
        cmd = f"carbs cluster {afa}"
        cmd_list.append(cmd)

    train_file = (
        os.path.join(carbs_output_directory, os.path.splitext(os.path.basename(afa))[0])
        + f".{pid}-train.fa"
    )
    if not os.path.isfile(train_file):
        # then split it, if you haven't already.
        cmd = f"carbs split -T argument --split_test --output_path {carbs_output_directory} {afa} {pid}"
        cmd_list.append(cmd)
    # use the domtblout file to get labels.
    domtblout = os.path.splitext(afa)[0] + ".domtblout"
    if domtblout_path is not None:
        domtblout = os.path.join(domtblout_path, os.path.basename(domtblout))

    if not os.path.isfile(domtblout):
        raise ValueError(f"couldn't find domtblout at {domtblout}, exiting.")

    domtblout_cmd = "grep \">\" {} | sed 's/>//g' | sed 's/ .*//g' | grep -f - {} | awk '{{print $1,$4,$5,$7}}' | label_fasta.py --output_directory {} --fasta_file {} -"
    # make the filenames for test, valid, and train
    test_file = (
        os.path.join(carbs_output_directory, os.path.splitext(os.path.basename(afa))[0])
        + f".{pid}-test.fa"
    )
    valid_file = (
        os.path.join(carbs_output_directory, os.path.splitext(os.path.basename(afa))[0])
        + f".{pid}-valid.fa"
    )

    # add labels into the clustered train file and save to the fasta output directory
    cmd = (
        f"if [[ -f {train_file} ]]; then\n"
        + domtblout_cmd.format(
            train_file, domtblout, fasta_output_directory, train_file
        )
        + "\nfi"
    )
    cmd_list.append(cmd)
    # add labels into the clustered test file and save to the fasta output directory
    cmd = (
        f"if [[ -f {test_file} ]]; then\n"
        + domtblout_cmd.format(test_file, domtblout, fasta_output_directory, test_file)
        + "\nfi"
    )
    cmd_list.append(cmd)
    # add labels into the clustered valid file and save to the fasta output directory
    cmd = (
        f"if [[ -f {valid_file} ]]; then\n"
        + domtblout_cmd.format(
            valid_file, domtblout, fasta_output_directory, valid_file
        )
        + "\nfi"
    )
    cmd_list.append(cmd)

    with open(slurm_outfile, "w") as dst:
        dst.write("\n\n".join(cmd_list))


if __name__ == "__main__":

    args = create_parser().parse_args()
    afa = args.aligned_fasta
    pid = args.pid
    carbs_output_directory = args.carbs_output_directory
    fasta_output_directory = args.output_directory
    slurm_out_directory = args.slurm_output_directory
    domtblout_path = args.domtblout_path

    if args.from_glob:
        afa_files = glob(os.path.join(os.path.dirname(afa), "*.afa"))
        for afa in afa_files:
            create_slurm_script(
                afa,
                pid,
                carbs_output_directory,
                fasta_output_directory,
                slurm_out_directory,
                domtblout_path,
            )
    else:
        create_slurm_script(
            afa,
            pid,
            carbs_output_directory,
            fasta_output_directory,
            slurm_out_directory,
            domtblout_path,
        )
