import json
import os
import shutil
import numpy as np

from glob import glob
from random import shuffle
from argparse import ArgumentParser


def parser():
    p = ArgumentParser()
    ap = p.add_subparsers(dest="command")
    clan_parser = ap.add_parser("clan", add_help=False)
    clan_parser.add_argument(
        "--afa-path", required=True, help="where are the alignments stored?"
    )
    clan_parser.add_argument("--out-path", required=True)
    clan_parser.add_argument(
        "--clan-to-family-mapping", type=str, default="../resources/clan_id_to_pfam.txt"
    )
    clan_parser.add_argument(
        "--accession-id-to-name", type=str, default="../resources/clan_name_to_pfam.txt"
    )
    clan_parser.add_argument("--n-sequences-per-family", type=int, required=True)
    clan_parser.add_argument("--fraction-of-families-in-clans", type=float, default=0.5)

    random_parser = ap.add_parser("random")
    random_parser.add_argument(
        "fa_path", help="where are the fast files stored?"
    )
    random_parser.add_argument("out_path")
    random_parser.add_argument("n_families", type=int)
    return p.parse_args()


def _parse_clan_to_family_file(clan_to_family_file):
    clan_to_family = {}

    with open(clan_to_family_file, "r") as src:
        for line in src.readlines():
            split_line = line.split(" ")
            clan_id = split_line[0]
            pfam_families = split_line[1:]
            pfam_families = [s.replace(";", "") for s in pfam_families]
            pfam_families = [s.replace("\n", "") for s in pfam_families]
            pfam_families = list(filter(len, pfam_families))
            pfam_families = list(
                filter(lambda x: True if "PF" in x else False, pfam_families)
            )

            if len(pfam_families):
                clan_to_family[clan_id] = pfam_families

    return clan_to_family


def _load_accession_id_to_name_mapping(path):
    id_to_name = {}
    with open(path, "r") as src:
        for line in src.readlines():
            accession_id, name = line.replace("\n", "").split("\t")
            id_to_name[accession_id] = name

    return id_to_name


def _make_clan_to_regular_name(clan_to_family, id_to_name):
    clan_to_regular_name = {}
    for clan, families in clan_to_family.items():
        names = []
        for family in families:
            if family in id_to_name:
                names.append(id_to_name[family])
            else:
                # print('{} not in id_to_name'.format(family))
                pass
        clan_to_regular_name[clan] = names

    return clan_to_regular_name


def copy_files(pfam_families, args):
    os.makedirs(args.out_path, exist_ok=True)

    for family in pfam_families:

        afa_pth = os.path.join(args.afa_path, family + ".afa")
        ddgm_pth = os.path.join(args.afa_path, family + ".ddgm")

        if os.path.isfile(afa_pth) and os.path.isfile(ddgm_pth):
            out_afa = os.path.join(args.out_path, family + ".afa")
            out_ddgm = os.path.join(args.out_path, family + ".ddgm")
            shutil.copyfile(afa_pth, out_afa)
            shutil.copyfile(ddgm_pth, out_ddgm)

        else:
            if not os.path.isfile(afa_pth):
                print(".afa does not exist for {} in {}".format(family, args.afa_path))
            elif not os.path.isfile(ddgm_pth):
                print(".ddgm does not exist for {} in {}".format(family, args.afa_path))
            else:
                pass

    return


def main(args):
    clan_to_family = _parse_clan_to_family_file(args.clan_to_family_mapping)
    id_to_name = _load_accession_id_to_name_mapping(args.accession_id_to_name)

    clan_to_regular_name = _make_clan_to_regular_name(clan_to_family, id_to_name)

    # choose n_clans clans
    clans = np.random.choice(
        list(clan_to_regular_name.keys()),
        size=len(clan_to_regular_name)
        if args.n_clans > len(clan_to_regular_name)
        else args.n_clans,
        replace=False,
    )

    pfam_families = [
        family for c in clans for family in clan_to_regular_name[c]
    ]  # flatten list
    copy_files(pfam_families, args)


def get_file_tuples(fpath):
    """
    Assumes the fpath is formatted like:
    /path/to/file/file.<percent-id>-{test,train,valid}.fa
    Returns a tuple (train, valid, test) of the matching files with None if the file doesn't exist.
    :param fpath: the file to match.
    :type fpath: str
    """
    bs = os.path.basename(fpath)
    dirname = os.path.dirname(fpath)
    if 'test' in bs:
        replace_str = 'test'
    elif 'train' in bs:
        replace_str = 'train'
    elif 'valid' in bs:
        replace_str = 'valid'
    else:
        raise ValueError(f"expected one of (test, train, valid) in the .fa file, got {fpath}")

    train_path = os.path.join(dirname, bs.replace(replace_str, "train"))
    test_path =  os.path.join(dirname, bs.replace(replace_str, "test"))
    valid_path = os.path.join(dirname, bs.replace(replace_str, "valid"))
    existing_files = []

    if os.path.isfile(train_path):
        existing_files.append(train_path)
    else:
        existing_files.append(None)

    if os.path.isfile(valid_path):
        existing_files.append(valid_path)
    else:
        existing_files.append(None)

    if os.path.isfile(test_path):
        existing_files.append(test_path)
    else:
        existing_files.append(None)

    return tuple(existing_files)


def grab_random_sets(directory, out_path, n):
    """
    Grabs random .fa files from directory if they have the same prefix and
    there are -valid and -train sets.
    :param n: number of .fa triplets to choose.
    :type n: int
    :param out_path: Where to save the selected data.
    :type out_path: str
    :param directory: Where the .fa files are stored.
    :type directory: str
    """
    os.makedirs(out_path, exist_ok=True)
    files = glob(os.path.join(directory, "*.fa"))
    shuffle(files)
    already_added = set()
    existing_files = []
    for f in files:
        train_file, valid_file, test_file = get_file_tuples(f)
        if valid_file is None or test_file is None:
            continue
        else:
            ftuple = (train_file, valid_file, test_file)
            if ftuple not in already_added:
                existing_files.append(ftuple)
                already_added.add(ftuple)
            if len(existing_files) == n:
                break

    for train_file, valid_file, test_file in existing_files:
        shutil.copy(train_file, out_path)
        shutil.copy(valid_file, out_path)
        shutil.copy(test_file, out_path)


if __name__ == "__main__":

    parser_args = parser()
    if parser_args.command == 'random':
        base_out_path = '/home/tc229954/data/prefilter/training_data/{}/{}/'
        base_data_path = '/home/tc229954/data/prefilter/clustered/{}/'
        pids = [0.2, 0.5, 0.35]
        n_seqs = [10000, 100, 2000, 500]
        for pid in pids:
            for n in n_seqs:
                out_path = base_out_path.format(pid, n)
                data_path = base_data_path.format(pid)
                print(out_path, data_path)
                grab_random_sets(data_path, out_path, n)
    elif parser_args.command == "clan":
        main(parser_args)
    else:
        print('whaat?')
