from src.data.utils import get_data_from_subset
import os
import pdb
import numpy as np
from src.evaluators.metrics import plot_roc_curve, recall_and_filtration
import matplotlib.pyplot as plt
import tqdm
# querysequences, targetsequences, all_hits = get_data_from_subset(
#     "uniref/phmmer_normal_results", query_id=query_filenum, file_num=target_filenum
# )

COLORS = ["r", "c", "g", "k"]
COLORS2 = ["r", "c", "g", "k"]
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

    # for color, evalue_threshold in zip(["r", "c", "g", "k"], [1e-10, 1e-1, 1, 10]):
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


            if recall is not None and filtration is not None:
                if idx % 3000 == 0:   
                    print(f"num_Ps: {num_positives},  num_Ds: {num_decoys},  recall: {100*recall:.3f}, filtration: {100*(1-filtration):.3f}")
                    filtrations.append(100*(1-filtration))
                    recalls.append(100*recall)
        datafile.close()
                #print(f"recall: {recall:.3f}, filtration: {filtration:.3f}")

       # ax.scatter(filtrations, recalls, c=COLORS[i], marker="o")
        ax.plot(filtrations, recalls, f"{COLORS[i]}--", linewidth=2, label=evalue_threshold)

    ax.plot([0, 100], [100, 0], "k--", linewidth=2)
    #ax.set_ylim([-1, 101])
    #ax.set_xlim([-1, 101])
    ax.set_xlabel("filtration")
    ax.set_ylabel("recall")
    plt.legend()
    plt.savefig(f"{figure_path}", bbox_inches="tight")
    plt.close()

def get_data(modelhitsfile):
    all_scores = []
    all_pairs = []
    for queryhits in os.listdir(modelhitsfile):
        queryname = queryhits.strip(".txt")
        if queryname not in all_hits_max.keys():
            print(f"Query {queryname} not in hmmer hits")
            continue
        
        file = open(f"{modelhitsfile}/{queryhits}", "r")
        scores = file.readlines()
        # file = open(f"{modelhitsfile}/{queryhits}", "r")
        # scores = file.read().replace('\n', '')
        # scores = scores.replace('UniRef90', '\n' + 'UniRef90')
        # scores = scores.split('\n')
        for line in scores:
            if "Distance" in line:
                continue
            target = line.split()[0].strip("\n")

            score = float(line.split()[1].strip("\n"))
            all_scores.append(score)

            all_pairs.append((queryname, target))
        file.close()
    return all_scores, all_pairs 


def write_datafile(pairs, evalue_thresholds = [1e-10,1e-1,1,10], filename = 'data.txt'):
    datafile = open(filename, 'w')
    numpos1 = numpos2 = numpos3 = numpos4 = 0
    for pair in pairs:
        query, target = pair[0], pair[1]

        evalue = None
        if target not in all_hits_max[query].keys():
            classname1 = classname2 = classname3 = classname4 = 'D'
        else:
            evalue = all_hits_max[query][target][0]
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

# filename = 'data0_distance_hits.txt'
# scores, pairs = get_data(distance_hits, filename=filename)
# print("Got data")

# sortedidx = np.argsort(scores)[::-1]
# pairs = np.array(pairs)[sortedidx]
# numhmmerhits = write_datafile(pairs)
# print("Wrote files")

query_filenum = 1
target_filenum =1

querysequences_max, targetsequences_max, all_hits_max = get_data_from_subset(
    "/xdisk/twheeler/daphnedemekas/phmmer_max_results",
    query_id=query_filenum,
    file_num=target_filenum,
)
# plot_roc_curve("ResNet1d/eval/roc_distances_max.png", numhmmerhits,filename=filename)
distance_hits = f"/xdisk/twheeler/daphnedemekas/prefilter-output/DistanceSums/{query_filenum}/{target_filenum}/distances"
distance_sum_hits = f"/xdisk/twheeler/daphnedemekas/prefilter-output/DistanceSums/{query_filenum}/{target_filenum}/distances_summed"

filename = 'data1_distance_sum_hits.txt'

scores, pairs = get_data(distance_sum_hits)
print("Got data")
sortedidx = np.argsort(scores)[::-1]
pairs = np.array(pairs)[sortedidx]
numpos1, numpos2, numpos3, numpos4 = write_datafile(pairs,evalue_thresholds = [1e-10,1e-1,1,10], filename=filename)
print("Wrote files")
plot_roc_curve("ResNet1d/eval/roc_distances_sums_filtered.png", numpos1, numpos2, numpos3, numpos4,filename=filename)
os.remove(filename)

filename = 'data1_distance_sum_hits.txt'

scores, pairs = get_data(distance_sum_hits)
print("Got data")
sortedidx = np.argsort(scores)[::-1]
pairs = np.array(pairs)[sortedidx]
numpos1, numpos2, numpos3, numpos4 = write_datafile(pairs,evalue_thresholds = [1e-10,1e-1,1,10], filename=filename)
print("Wrote files")
plot_roc_curve("ResNet1d/eval/roc_distances_max_filtered.png", numpos1, numpos2, numpos3, numpos4,filename=filename)
os.remove(filename)

raise




# all_distances_full = np.load('all_distances_full.npy')
# all_e_values2 = np.load('all_e_values2.npy')
# all_targets = np.load('all_targets.npy')
# all_bias = np.load('all_biases.npy')
# numhits = len(all_targets)


# d_idxs = np.where(all_distances_full > 1000)[0]

# e_vals = all_e_values2[d_idxs]

# outliers = np.where(e_vals > 1)[0]

# outlier_idx = d_idxs[outliers]

# pairs = all_targets[outlier_idx]

# f = open('outliers.txt', 'w')
# for idx in outlier_idx:
#     pair = all_targets[idx]
#     f.write('Query' + '\n' + str(pair[0])+ '\n')
#     f.write(querysequences_max[pair[0]]+ '\n')
#     f.write('Target' + '\n' + str(pair[1])+ '\n')
#     f.write(targetsequences_max[pair[1]]+ '\n')

#     f.write('Predicted Similarity: ' + str(all_distances_full[idx]) + '\n')
#     f.write('E-value: '  + str(all_e_values2[idx]) + '\n')
# f.close()

# loge = np.ma.masked_invalid(np.log10(all_e_values2))
# idxs = np.where(loge < -250)[0]

# f = open('inliers.txt', 'w')
# for idx in idxs:
#     pair = all_targets[idx]
#     f.write('Query' + '\n' + str(pair[0])+ '\n')
#     f.write(querysequences_max[pair[0]]+ '\n')
#     f.write('Target' + '\n' + str(pair[1])+ '\n')
#     f.write(targetsequences_max[pair[1]]+ '\n')

#     f.write('Predicted Similarity: ' + str(all_distances_full[idx]) + '\n')
#     f.write('E-value: '  + str(all_e_values2[idx]) + '\n')
# f.close()

#distance_hits_dict = {}
distance_sum_hits_dict = {}

# all_distances = []
# all_e_values = []
# all_scores = []
all_distances2 = []
all_distances_full = []

all_e_values2 = []
all_bias2 = []
targets = []
all_targets = []
for queryhits in os.listdir(distance_hits):
    queryname = queryhits.strip(".txt")
    individual_distances = open(f"{distance_hits}/{queryhits}", "r")
    with open(f"{distance_sum_hits}/{queryhits}", "r") as file:
        file = file.read().replace('\n', '')
        summed_distances = file.replace('UniRef90', '\n' + 'UniRef90')
    summed_distances = summed_distances.split('\n')
    if queryname not in all_hits_max.keys():
        continue
    # distance_hits_dict[queryname] = {}
    distance_sum_hits_dict[queryname] = {}

    # for line in individual_distances:
    #     if "Distance" in line:
    #         continue
    #     target = line.split()[0].strip("\n")
    #     if target not in all_hits_max[queryname].keys():
    #         continue
    #     if target not in targets:
    #         targets.append(target)
    #     distance = float(line.split()[1].strip("\n"))

    #     distance_hits_dict[queryname][target] = distance

    #     all_distances.append(distance)
    #     all_e_values.append(all_hits_max[queryname][target][0])
    #     all_scores.append(all_hits_max[queryname][target][1])
    
    
    
    for line in summed_distances:
        if "Distance" in line:
            continue
        try:
            target = line.split()[0].strip("\n")
            if target not in all_hits_max[queryname].keys():
                continue
            if target not in targets:
                targets.append(target)
            distance = float(line.split()[1].strip("\n"))
            all_distances_full.append(distance)

            #target_length = len(targetsequences_max[target])
            #distance = distance / target_length
            #all_distances2.append(distance)

            # print(f"Target length: {target_length}")
            # print(f"Distance: {distance}")
            # print(f"Distance / target length: {distance/target_length}")

            distance_sum_hits_dict[queryname][target] = distance

            all_e_values2.append(all_hits_max[queryname][target][0])
            all_bias2.append(all_hits_max[queryname][target][2])
            all_targets.append((queryname, target))
        except:
            continue

all_distances_full = np.array(all_distances_full)
#all_distances2 = np.array(all_distances2)
#all_distances = np.array(all_distances)
all_e_values2 = np.array(all_e_values2)
#all_e_values = np.array(all_e_values)
all_bias2 = np.array(all_bias2)
pdb.set_trace()

plot_roc_curve( distance_sum_hits_dict, all_hits_max,True,0,len(all_targets),"ResNet1d/eval/roc.png",np.greater_equal,evalue_thresholds=[1e-10, 1e-1, 1, 10],meanvalue = np.mean(all_distances_full)+np.std(all_distances_full)*2, maxvalue = np.max(all_distances_full))
plt.clf()


np.save('all_distances_full', all_distances_full, allow_pickle = True)
np.save('all_biases', all_bias2, allow_pickle = True)

#np.save('all_distances2', all_distances2, allow_pickle = True)
#np.save('all_distances', all_distances, allow_pickle = True)
np.save('all_e_values2', all_e_values2, allow_pickle = True)
#np.save('all_e_values', all_e_values, allow_pickle = True)



numhits = np.sum([len(distance_sum_hits_dict[q]) for q in list(distance_sum_hits_dict.keys())])
print(
    f"Got {numhits} total hits from our model"
)
pdb.set_trace()
raise
over_7000 = np.where(np.array(all_distances_full) > 7000)[0]

e_val_7000 = np.array(all_e_values2)[over_7000]



#plot_mean_e_values(all_distances2, all_e_values2, fn = 'evaluemeans_summed', max = 300)
#plot_mean_e_values(all_distances, all_e_values, fn = 'evaluemeans',max = 1)

# plot_mean_e_values(all_distances2, all_e_values2, fn = 'evaluemeans_summed', max = 1e-1)
# plot_mean_e_values(all_distances, all_e_values, max = 1e-1)

# plot_mean_e_values(all_distances2, all_e_values2, fn = 'evaluemeans_summed', max = 1)
# plot_mean_e_values(all_distances, all_e_values, max = 1)


# plot_mean_e_values(all_distances2, all_e_values2, fn = 'evaluemeans_summed', max = 10)
# plot_mean_e_values(all_distances, all_e_values, max = 10)
#plot_both_roc(distance_hits_dict, distance_sum_hits_dict,all_hits_max,True,0,numhits,"ResNet1d/eval/summed_distances.png",np.greater_equal,evalue_thresholds=[1e-10, 1e-1, 1, 10],maxvalue = np.max(all_distances2))
pdb.set_trace()

