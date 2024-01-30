import argparse
import pickle
import tqdm
from src.data.hmmerhits import FastaFile


def get_numpos_per_evalue(hmmer_hits, query_sequences, target_sequences):
    evalue_thresholds = [1e-10, 1e-5, 1e-1, 10]
    num_pos_per_evalue = [0, 0, 0, 0]
    num_marg_per_evalue = [0, 0, 0, 0]
    num_decoys = [0, 0, 0, 0]

    numhits = len(query_sequences) * len(target_sequences)
    for query in tqdm.tqdm(query_sequences):
        if query not in hmmer_hits:
            continue

        query_data = hmmer_hits[query]
        for target in target_sequences:
            if target not in query_data:
                continue
            data = query_data[target]
            evalue = data[0]
            if evalue < evalue_thresholds[3]:  # evalue < 10
                num_pos_per_evalue[3] += 1
                if evalue < evalue_thresholds[2]:  # evalue < e-1
                    num_pos_per_evalue[2] += 1

                    if evalue < evalue_thresholds[1]:
                        num_pos_per_evalue[1] += 1

                        if evalue < evalue_thresholds[0]:
                            num_pos_per_evalue[0] += 1
                        else:
                            num_marg_per_evalue[0] += 1
                    else:
                        num_marg_per_evalue[0] += 1
                        num_marg_per_evalue[1] += 1

                else:  # e-1 < evalue < 10
                    num_marg_per_evalue[0] += 1
                    num_marg_per_evalue[1] += 1
                    num_marg_per_evalue[2] += 1
            else:
                num_marg_per_evalue[0] += 1
                num_marg_per_evalue[1] += 1
                num_marg_per_evalue[2] += 1
                num_marg_per_evalue[3] += 1

    for i in range(4):
        num_decoys[i] = numhits - num_pos_per_evalue[i] - num_marg_per_evalue[i]
    return num_pos_per_evalue, num_decoys


def main(hmmer_hits_file, targets_file, queries_file, results_file):
    with open(hmmer_hits_file + ".pkl", "rb") as file:
        all_hits_max = pickle.load(file)

    targetfile = FastaFile(targets_file)
    queriesfile = FastaFile(queries_file)

    query_sequences = queriesfile.data
    target_sequences = targetfile.data

    num_pos_per_evalue, numdecoys = get_numpos_per_evalue(
        all_hits_max, query_sequences, target_sequences
    )

    print(f"Num pos per evalue: {num_pos_per_evalue}")
    print(f"Num decoys: {numdecoys}")

    with open(results_file, "w") as f:
        f.write(f"HMMER MAX pos: {num_pos_per_evalue}" + "\n")
        f.write(f"HMMER MAX decoys: {numdecoys}")

    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hmmer_hits_file", type=str, help="Path to the hmmer hits file"
    )
    parser.add_argument("--targets_file", type=str, help="Path to the targets file")
    parser.add_argument("--queries_file", type=str, help="Path to the queries file")
    parser.add_argument(
        "--results_file", type=str, help="Path where to save the results"
    )

    args = parser.parse_args()
    main(args.hmmer_hits_file, args.targets_file, args.queries_file, args.results_file)
