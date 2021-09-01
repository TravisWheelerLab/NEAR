import json
import os
import shutil
import numpy as np

from glob import glob
from random import shuffle
from argparse import ArgumentParser


def parser():
    ap = ArgumentParser()
    ap.add_argument('--clan-to-family-mapping', required=True)
    ap.add_argument('--accession-id-to-name', required=True)
    ap.add_argument('--afa-path', required=True)
    ap.add_argument('--out-path', required=True)
    ap.add_argument('--n-clans', type=int, default=100)

    return ap.parse_args()


def _parse_clan_to_family_file(clan_to_family_file):
    clan_to_family = {}

    with open(clan_to_family_file, 'r') as src:
        for line in src.readlines():
            split_line = line.split(' ')
            clan_id = split_line[0]
            pfam_families = split_line[1:]
            pfam_families = [s.replace(';', '') for s in pfam_families]
            pfam_families = [s.replace('\n', '') for s in pfam_families]
            pfam_families = list(filter(len, pfam_families))
            pfam_families = list(filter(lambda x: True if 'PF' in x else False, pfam_families))

            if len(pfam_families):
                clan_to_family[clan_id] = pfam_families

    return clan_to_family


def _load_accession_id_to_name_mapping(path):
    id_to_name = {}
    with open(path, 'r') as src:
        for line in src.readlines():
            accession_id, name = line.replace('\n', '').split("\t")
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
            print('either .ddgm or .afa does not exist for {} in {}'.format(family, args.afa_path))

    return


def main(args):
    clan_to_family = _parse_clan_to_family_file(args.clan_to_family_mapping)
    id_to_name = _load_accession_id_to_name_mapping(args.accession_id_to_name)

    clan_to_regular_name = _make_clan_to_regular_name(clan_to_family, id_to_name)

    # choose n_clans clans
    clans = np.random.choice(list(clan_to_regular_name.keys()), size=len(clan_to_regular_name) if args.n_clans > len(clan_to_regular_name) else args.n_clans, replace=False)

    pfam_families = [family for c in clans for family in clan_to_regular_name[c]]  # flatten list
    copy_files(pfam_families, args)


if __name__ == '__main__':

    parser_args = parser()

    main(parser_args)
