import os
import tqdm
import argparse
import itertools


def get_training_data(path):

    files = os.listdir(path)

    training_data = {}
    for file in tqdm.tqdm(files):
        f = open(f"{path}/{file}", "r")
        text = f.readlines()
        f.close()
        if len(text) == 0:
            print(f"File {f'{path}/{file}'} is empty.")
            os.remove(f"{path}/{file}")
            continue

        query_and_target = text[0]
        query = query_and_target.split()[0].strip(">").strip("\n")
        target = query_and_target.split()[-1].strip("\n")
        queryseq = text[1].strip("\n")
        targetseq = text[2].strip("\n")

        fullquery = text[3].strip("\n")
        fulltarget = text[4].strip("\n")

        if not query in training_data.keys():
            training_data[query] = {}

        training_data[query][queryseq] = {
            "target": target,
            "targetsequence": targetseq,
            "fullqseq": fullquery,
            "fulltseq": fulltarget,
        }

    return training_data


def write_new_data_with_matches(new_training_data_path, training_data):

    i = 0
    if len(os.listdir(new_training_data_path)) != 0:
        i = max([int(f.strip(".txt")) for f in os.listdir(new_training_data_path)])
    print("Writing training data...")
    for query in tqdm.tqdm(list(training_data.keys())[i:]):
        sequences = training_data[query].keys()
        for sequence in sequences:
            querymatches = [
                (s, s.find(sequence)) for s in sequences if sequence in s and sequence != s
            ]
            querytarget = training_data[query][sequence]
            if len(querymatches) > 0:
                targetmatches = []
                f = open(f"{new_training_data_path}/{i}.txt", "w")
                line1 = f'{query} & {querytarget["target"]} '
                for match in querymatches:
                    seq, idx = match
                    targetmatch = training_data[query][seq].copy()
                    targetmatch.update(
                        {
                            "targetsequence": training_data.copy()[query][seq]["targetsequence"][
                                idx : idx + len(sequence)
                            ]
                        }
                    )
                    targetmatches.append(targetmatch)

                for t in targetmatches:
                    line1 += f'& {t["target"]}'
                f.write(line1 + "\n")
                f.write("<" + sequence + "\n")
                f.write("<" + querytarget["targetsequence"] + "\n")
                for t in targetmatches:
                    f.write("<" + t["targetsequence"] + "\n")
                f.write(querytarget["fullqseq"] + "\n")
                f.write(querytarget["fulltseq"] + "\n")
                for t in targetmatches:
                    f.write(t["fulltseq"] + "\n")
                f.close()
            else:
                f = open(f"{new_training_data_path}/{i}.txt", "w")
                f.write(f'{query} & {querytarget["target"]} ' + "\n")
                f.write("<" + sequence + "\n")
                f.write("<" + querytarget["targetsequence"] + "\n")
                f.write(querytarget["fullqseq"] + "\n")
                f.write(querytarget["fulltseq"] + "\n")
                f.close()
            i += 1


def main_train(single_pos_training_data: str, multi_pos_training_data: str):
    print(f"Writing multi-positives to {multi_pos_training_data}")
    training_data = get_training_data(single_pos_training_data)
    write_new_data_with_matches(multi_pos_training_data, training_data)


def main_eval(single_pos_eval_data: str, multi_pos_eval_data: str):
    print(f"Writing multi-positives to {multi_pos_eval_data}")

    eval_data = get_training_data(single_pos_eval_data)
    write_new_data_with_matches(multi_pos_eval_data, eval_data)


parser = argparse.ArgumentParser()
parser.add_argument("single_pos_training_data")
parser.add_argument("multi_pos_training_data")
parser.add_argument("single_pos_eval_data")
parser.add_argument("multi_pos_eval_data")
args = parser.parse_args()


main_train(args.single_pos_training_data, args.multi_pos_training_data)
main_eval(args.single_pos_eval_data, args.multi_pos_eval_data)
