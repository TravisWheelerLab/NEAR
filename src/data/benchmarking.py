from src.data.utils import get_data_from_subset, get_evaluation_data
import os
import pdb
import numpy as np
from src.evaluators.metrics import plot_roc_curve
import matplotlib.pyplot as plt
import tqdm


COLORS = ["r", "c", "g", "k"]

def plot_mean_e_values(all_distance, all_e_values, biases, min = 0, max = 300, alpha = 0.5, fn = 'evaluemeans', plot_stds = True, plot_lengths = True, title = '', s = 1):
    plt.clf()
    thresholds = np.linspace(min,max,100)
    all_distance = np.array(all_distance)
    all_e_values = np.array(all_e_values)
    
    means = []
    stds = []
    lengths = []
    mean_bias = []
    for threshold in thresholds:
        idx = np.where(all_distance>threshold)[0]
        mean = np.mean(np.ma.masked_invalid(np.log10(all_e_values[idx])))
        std = np.std(np.ma.masked_invalid(np.log10(all_e_values[idx])))
        bias = np.mean(biases[idx])
        means.append(mean)
        stds.append(std)
        length = len(idx)
        lengths.append(length)
        mean_bias.append(bias)
    lengths = np.log(lengths)
    plt.scatter(thresholds, means, c = mean_bias, cmap = 'Greens', s = s)
    plt.plot(thresholds, means)
    if plot_stds:
        plt.fill_between(thresholds, np.array(means) - np.array(stds)/2, np.array(means) + np.array(stds)/2, alpha = alpha)
    if plot_lengths:
        plt.fill_between(thresholds, np.array(means) - np.array(lengths), np.array(means) + np.array(lengths), alpha = 0.5, color = 'orange')
    plt.title(title)
    plt.ylabel("Log E value means")
    plt.xlabel("Similarity Threshold")
    plt.savefig(f"ResNet1d/eval/{fn}.png")

def plot_roc_curve(figure_path,numpos1, numpos2, numpos3, numpos4, evalue_thresholds=[1e-10,1e-1, 1, 10], filename = 'data.txt'
):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f"{os.path.splitext(os.path.basename(figure_path))[0]}")
    datafile = open(filename,'r')
    lines = datafile.readlines()
    numhits = len(lines)
    numpos = [numpos1,numpos2,numpos3,numpos4]

    for i, evalue_threshold in enumerate(evalue_thresholds):
        print(f'Evalue threshold: {evalue_threshold}')

        filtrations = []
        recalls = []

        num_positives = 0
        num_decoys = 0
        recall = None
        filtration = None

        for idx, line in enumerate(lines):
            line = line.split()
            classname = line[3+i]
            if classname == 'P':
                num_positives +=1 
                recall = num_positives / numpos[i]
            elif classname == 'D':
                num_decoys += 1
                filtration = num_decoys / numhits


            if recall is not None and filtration is not None and (1-filtration) > 0.8:
                if idx % 1000 == 0:   
                    print(f"num_Ps: {num_positives},  num_Ds: {num_decoys},  recall: {100*recall:.3f}, filtration: {100*(1-filtration):.3f}")
                    filtrations.append(100*(1-filtration))
                    recalls.append(100*recall)
        datafile.close()

        ax.plot(filtrations, recalls, f"{COLORS[i]}--", linewidth=2, label=evalue_threshold)

    ax.plot([0, 100], [100, 0], "k--", linewidth=2)

    ax.set_xlabel("filtration")
    ax.set_ylabel("recall")
    plt.legend()
    plt.savefig(f"{figure_path}", bbox_inches="tight")
    plt.close()

def get_scores_and_pairs(modelhitsfile, hmmerhits):
    all_scores = []
    all_pairs = []
    for queryhits in os.listdir(modelhitsfile):
        queryname = queryhits.strip(".txt")
        if queryname not in hmmerhits.keys():
            print(f"Query {queryname} not in hmmer hits")
            continue
        
        file = open(f"{modelhitsfile}/{queryhits}", "r")
        scores = file.readlines()

        for line in scores:
            if "Distance" in line:
                continue
            target = line.split()[0].strip("\n")

            score = float(line.split()[1].strip("\n"))
            all_scores.append(score)

            all_pairs.append((queryname, target))
        file.close()
    return all_scores, all_pairs 


def write_datafile(pairs,hmmerhits, evalue_thresholds = [1e-10,1e-1,1,10], filename = 'data.txt'):
    datafile = open(filename, 'w')
    numpos1 = numpos2 = numpos3 = numpos4 = 0
    for pair in pairs:
        query, target = pair[0], pair[1]

        evalue = None
        if target not in hmmerhits[query].keys():
            classname1 = classname2 = classname3 = classname4 = 'D'
        else:
            evalue = hmmerhits[query][target][0]
            classname1 = classname2 = classname3 = classname4 = 'M'

            if evalue < evalue_thresholds[3]: 
                classname4 = 'P'
                numpos4 += 1
                if evalue < evalue_thresholds[2]: 
                    classname3 = 'P'
                    numpos3 += 1
                    if evalue < evalue_thresholds[1]: 
                        classname2 = 'P'
                        numpos2 += 1
                        if evalue < evalue_thresholds[0]: 
                            classname1 = 'P'
                            numpos1 += 1

        datafile.write(f'{query}          {target}          {evalue}          {classname1}          {classname2}          {classname3}          {classname4}' + '\n')
    datafile.close()

    return numpos1, numpos2, numpos3, numpos4


def generate_roc(filename, modelhitsfile, hmmerhits, figure_path):
    scores, pairs = get_scores_and_pairs(modelhitsfile, hmmerhits)
    print("Got data")
    sortedidx = np.argsort(scores)[::-1]
    pairs = np.array(pairs)[sortedidx]
    numpos1, numpos2, numpos3, numpos4 = write_datafile(pairs,hmmerhits,evalue_thresholds = [1e-10,1e-1,1,10], filename=filename)
    print("Wrote files")
    plot_roc_curve(figure_path, numpos1, numpos2, numpos3, numpos4,filename=filename)
    os.remove(filename)


def get_outliers_and_inliers(all_similarities, all_e_values, all_targets):
    d_idxs = np.where(all_similarities > 1000)[0]

    e_vals = all_e_values[d_idxs]

    outliers = np.where(e_vals > 1)[0]

    outlier_idx = d_idxs[outliers]

    f = open('outliers.txt', 'w')
    for idx in outlier_idx:
        pair = all_targets[idx]
        f.write('Query' + '\n' + str(pair[0])+ '\n')
        f.write(querysequences_max[pair[0]]+ '\n')
        f.write('Target' + '\n' + str(pair[1])+ '\n')
        f.write(targetsequences_max[pair[1]]+ '\n')

        f.write('Predicted Similarity: ' + str(all_similarities[idx]) + '\n')
        f.write('E-value: '  + str(all_e_values[idx]) + '\n')
    f.close()

    loge = np.ma.masked_invalid(np.log10(all_e_values))
    idxs = np.where(loge < -250)[0]

    f = open('inliers.txt', 'w')
    for idx in idxs:
        pair = all_targets[idx]
        f.write('Query' + '\n' + str(pair[0])+ '\n')
        f.write(querysequences_max[pair[0]]+ '\n')
        f.write('Target' + '\n' + str(pair[1])+ '\n')
        f.write(targetsequences_max[pair[1]]+ '\n')

        f.write('Predicted Similarity: ' + str(all_similarities[idx]) + '\n')
        f.write('E-value: '  + str(all_e_values[idx]) + '\n')
    f.close()


def get_data(hits_path):
    similarity_hits_dict = {}
    all_similarities = []

    all_e_values = []
    all_biases = []
    all_targets = []
    for queryhits in os.listdir(hits_path):
        queryname = queryhits.strip(".txt")
        similarities = open(f"{hits_path}/{queryhits}", "r")
        similarities = similarities.readlines()
        if queryname not in all_hits_max.keys():
            continue
        similarity_hits_dict[queryname] = {}
        
        for line in similarities:
            if "Distance" in line:
                continue
            try:
                target = line.split()[0].strip("\n")
                if target not in all_hits_max[queryname].keys():
                    continue
                similarity = float(line.split()[1].strip("\n"))
                all_similarities.append(similarity)

                similarity_hits_dict[queryname][target] = similarity

                all_e_values.append(all_hits_max[queryname][target][0])
                all_biases.append(all_hits_max[queryname][target][2])
                all_targets.append((queryname, target))
            except:
                continue

    all_similarities = np.array(all_similarities)
    all_e_values = np.array(all_e_values)
    all_biases = np.array(all_biases)

    np.save('all_similarities', all_similarities, allow_pickle = True)
    np.save('all_biases', all_biases, allow_pickle = True)
    np.save('all_e_values', all_e_values, allow_pickle = True)


    numhits = np.sum([len(similarity_hits_dict[q]) for q in list(similarity_hits_dict.keys())])
    print(
        f"Got {numhits} total hits from our model"
    )
    pdb.set_trace()

# query_filenum = 4


# target_filenums = range(45)

# querysequences_max, targetsequences_max, all_hits_max = get_evaluation_data(
#     "/xdisk/twheeler/daphnedemekas/phmmer_max_results",
#     query_id=query_filenum,
# )

# hits_path = f"/xdisk/twheeler/daphnedemekas/prefilter-output/DistanceSums/{query_filenum}/{target_filenum}/distances_summed"

# filename = 'data1_distance_sum_hits.txt'

# get_data(hits_path)
# all_similarities = np.load('all_similarities.npy')
# all_e_values = np.load('all_e_values.npy')
# all_biases = np.load('all_biases.npy')
