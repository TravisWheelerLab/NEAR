import os 
import pickle 
import tqdm
all_hits_max_file_4 = "data/evaluationtargetdict"
all_hits_normal_file_4 = "data/evaluationtargetdictnormal"


def load_hmmer_hits(query_id: int = 4):
    """Loads pre-saved hmmer hits dictionaries for a given
    evaluation query id, currently can only be 4 or 0"""
    if query_id == 4:
        with open(all_hits_max_file_4 + ".pkl", "rb") as file:
            all_hits_max_4 = pickle.load(file)
        with open(all_hits_normal_file_4 + ".pkl", "rb") as file:
            all_hits_normal_4 = pickle.load(file)
        return all_hits_max_4, all_hits_normal_4
    else:
        raise Exception(f"No evaluation data for given query id {query_id}")


def parse_mmseqs():
    querynum = 4

    evalseqs = '/xdisk/twheeler/daphnedemekas/targetdataseqs/eval.txt'


    with open(evalseqs, "r") as val_target_file:
        val_targets = [t.strip("\n") for t in val_target_file.readlines()]
        print(f"Found {len(val_targets)} val targets")

    directory = '/xdisk/twheeler/daphnedemekas/mmseqs-output/4'

    mmseqs_output = {}

    outputdir = '/xdisk/twheeler/daphnedemekas/prefilter-output/mmseqs'
    os.mkdir(outputdir)

    for subdir in tqdm.tqdm(os.listdir(directory)):
        mmseqspath = f'{directory}/{subdir}/alnRes.m8'
        with open(mmseqspath, 'r') as f:
            lines = f.readlines()
            for f in tqdm.tqdm(lines):
                line_split = f.split()
                query, target = line_split[0], line_split[1]
                bitscore = float(line_split[-1])
                if os.path.exists(f"{outputdir}/{query}.txt"):
                    with open(f"{outputdir}/{query}.txt", "a") as f:
                    
                        f.write(f"{target}     {bitscore}" + "\n")
                else:
                    with open(f"{outputdir}/{query}.txt", "w") as f:
                    
                        f.write(f"{target}     {bitscore}" + "\n")      


def get_mmseqs_recall(e_value_threshold):
    all_hits_max_4, all_hits_normal_4 = load_hmmer_hits()
    with open('/xdisk/twheeler/daphnedemekas/mmseqs_hits.pkl', "rb") as file:
        mmseqs_hits = pickle.load(file)
    
    for query, target_list in mmseqs_hits.items():
        if query not in all_hits_max_4:
            print(f"Query {query} not in HMMER Max")
            num_decoys += len(target_list)
        else:
            hmmer_query_hits = all_hits_max_4[query]
            if target in hmmer_query_hits:
                e_value = target[0]


if __name__ == '__main__':
    parse_mmseqs()