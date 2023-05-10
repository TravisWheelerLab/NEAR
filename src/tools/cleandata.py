from src.datasets.alignmentgenerator import AlignmentGeneratorWithIndels
from src.data.hmmerhits import FastaFile
import tqdm
import pdb
import os

trainpaths = '/xdisk/twheeler/daphnedemekas/train_paths2.txt'
trainpathsclean = open('/xdisk/twheeler/daphnedemekas/train_paths_clean.txt','w')

fdataset = AlignmentGeneratorWithIndels(trainpaths, 512)

file = open(trainpaths,'r')
allines = file.readlines()
file.close()

INDEX = 0

for idx in tqdm.tqdm(range(INDEX, len(allines))):
    try:
        item = fdataset.__getitem__(idx)
        trainpathsclean.write(allines[idx])
        #print(idx)
    except Exception as e:
        if 'is empty' in str(e):
            print("Deleting empty file")
            os.remove(allines[idx].strip("\n"))
            continue
        # elif 'less than 4' in str(e):
        #     print("Writing full seqs")

        #     path = allines[idx].strip("\n")
        #     querynum = path.split('/')[-3]
        #     targetnum = path.split('/')[-2]
        #     if querynum != QUERYNUM:
        #         QUERYNUM = querynum
        #         queryfasta = FastaFile(f"uniref/split_subset/queries/queries_{QUERYNUM}.fa")
        #         querysequences = queryfasta.data
        #     if targetnum != TARGETNUM:
        #         TARGETNUM = targetnum
        #         targetfasta = FastaFile(f"uniref/split_subset/targets/targets_{TARGETNUM}.fa")
        #         targetsequences = targetfasta.data
        #     with open(path,'r') as file:
        #         lines = file.readlines()
        #         query_and_target = lines[0]
        #         query = query_and_target.split()[0].strip(">")
        #         target = query_and_target.split()[-1]
        #         if query not in querysequences or target not in targetsequences:
        #             pdb.set_trace()
        #         else:
        #             print("working")
        #         query_sequence = querysequences[query]
        #         target_sequence = targetsequences[target]
        #         print(query_sequence)
        #         print(target_sequence)
        #     with open(path,'a') as file:
        #         file.write("\n" + query_sequence + "\n")
        #         file.write(target_sequence + "\n")
        #     trainpathsclean.write(allines[idx])
        #     print("Done")
