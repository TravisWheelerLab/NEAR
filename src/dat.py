from src.data.utils import get_data_from_subset
import os
import pdb
import numpy as np
from src.evaluators.metrics import plot_roc_curve, recall_and_filtration
import matplotlib.pyplot as plt
import tqdm

query_filenum = 1
target_filenum = 0

querysequences_max, targetsequences_max, all_hits_max = get_data_from_subset(
    "/xdisk/twheeler/daphnedemekas/phmmer_max_results",
    query_id=query_filenum,
    file_num=target_filenum,
)
COLORS = ["r", "c", "g", "k"]
COLORS2 = ["r", "c", "g", "k"]


def plot_both_roc(
    distance_hits,
    distance_sum_hits,
    max_hmmer_hits,
    normalize_embeddings,
    distance_threshold,
    denom,
    figure_path,
    comp_func,
    evalue_thresholds=[1e-10, 1e-1, 1, 10],
    maxvalue = 0.99
):
    """Roc Curve for comparing model hits to the HMMER hits without the prefilter"""
    sum_thresholds = np.linspace(distance_threshold, maxvalue+10, num=10)
    distance_thresholds = np.append(np.linspace(0,0.5,7), np.linspace(0.6,1,3))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f"{os.path.splitext(os.path.basename(figure_path))[0]}")

    # for color, evalue_threshold in zip(["r", "c", "g", "k"], [1e-10, 1e-1, 1, 10]):
    for i, evalue_threshold in enumerate(evalue_thresholds):

        filtrations = []
        recalls = []
        for threshold in tqdm.tqdm(distance_thresholds):
            recall, total_hits = recall_and_filtration(
                distance_hits,
                max_hmmer_hits,
                threshold,
                comp_func,
                evalue_threshold,
            )

            filtration = 100 * (1.0 - (total_hits / denom))
            filtrations.append(filtration)
            recalls.append(recall)
            print(f"recall: {recall:.3f}, filtration: {filtration:.3f}, threshold: {threshold:.3f}")




        filtrations2 = []
        recalls2 = []
        for threshold in tqdm.tqdm(sum_thresholds):
            recall, total_hits = recall_and_filtration(
                distance_sum_hits,
                max_hmmer_hits,
                threshold,
                comp_func,
                evalue_threshold,
            )

            filtration = 100 * (1.0 - (total_hits / denom))
            filtrations2.append(filtration)
            recalls2.append(recall)
            print(f"recall: {recall:.3f}, filtration: {filtration:.3f}, threshold: {threshold:.3f}")

        ax.scatter(filtrations, recalls, color = 'mediumseagreen', marker="o")
        ax.scatter(filtrations2, recalls2, color = 'purple', marker="o")

        ax.plot(filtrations, recalls,  color = 'mediumseagreen', linewidth=2, label='distances')
        ax.plot(filtrations2, recalls2,  color = 'purple', linestyle = '--', linewidth=2, label='distances summed')
        plt.title(evalue_threshold)

        #ax.plot([0, 100], [100, 0], "k--", linewidth=2)
        ax.set_ylim([-1, 101])
        ax.set_xlim([-1, 101])
        ax.set_xlabel("filtration")
        ax.set_ylabel("recall")
        plt.legend()
        plt.savefig(f"ResNet1d/eval/{evalue_threshold}.png", bbox_inches="tight")
        plt.clf()


# querysequences, targetsequences, all_hits = get_data_from_subset(
#     "uniref/phmmer_normal_results", query_id=query_filenum, file_num=target_filenum
# )

# print('Max Hmmer Hits')
# print(len(all_hits_max))
# print('Hmmer hits')
# print(len(all_hits))

# if len(all_hits) > len(all_hits_max):
# 	x = all_hits
# 	all_hits = all_hits_max
# 	all_hits_max = x
# for query, tdict in all_hits.items():
# 	assert query in all_hits_max.keys()

#ourhits = f"/xdisk/twheeler/daphnedemekas/prefilter-output/ResNet1d/{query_filenum}/{target_filenum}/output"


distance_hits = f"/xdisk/twheeler/daphnedemekas/prefilter-output/DistanceSums/{query_filenum}/{target_filenum}/distances"
distance_sum_hits = f"/xdisk/twheeler/daphnedemekas/prefilter-output/DistanceSums/{query_filenum}/{target_filenum}/distances_summed"


print(len(querysequences_max))
print(len(targetsequences_max))
print(len(all_hits_max))




distance_hits_dict = {}
distance_sum_hits_dict = {}

all_distances = []
all_e_values = []
all_scores = []
all_distances2 = []

all_e_values2 = []
all_scores2 = []
targets = []

for queryhits in os.listdir(distance_hits):
    queryname = queryhits.strip(".txt")
    individual_distances = open(f"{distance_hits}/{queryhits}", "r")
    with open(f"{distance_sum_hits}/{queryhits}", "r") as file:
        file = file.read().replace('\n', '')
        summed_distances = file.replace('UniRef90', '\n' + 'UniRef90')
    summed_distances = summed_distances.split('\n')
    if queryname not in all_hits_max.keys():
        continue
    distance_hits_dict[queryname] = {}
    distance_sum_hits_dict[queryname] = {}

    for line in individual_distances:
        if "Distance" in line:
            continue
        target = line.split()[0].strip("\n")
        if target not in all_hits_max[queryname].keys():
            continue
        if target not in targets:
            targets.append(target)
        distance = float(line.split()[1].strip("\n"))

        distance_hits_dict[queryname][target] = distance

        all_distances.append(distance)
        all_e_values.append(all_hits_max[queryname][target][0])
        all_scores.append(all_hits_max[queryname][target][1])
    
    
    
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
            all_distances2.append(distance)

            distance = min(distance, 100)

            # target_length = len(targetsequences_max[target])

            # print(f"Target length: {target_length}")
            # print(f"Distance: {distance}")
            # print(f"Distance / target length: {distance/target_length}")

            distance_sum_hits_dict[queryname][target] = distance

            all_e_values2.append(all_hits_max[queryname][target][0])
            all_scores2.append(all_hits_max[queryname][target][1])
        except:
            continue


numhits = np.sum([len(distance_sum_hits_dict[q]) for q in list(distance_sum_hits_dict.keys())])
print(
    f"Got {numhits} total hits from our model"
)
plot_both_roc(distance_hits_dict, distance_sum_hits_dict,all_hits_max,True,0,numhits,"ResNet1d/eval/summed_distances.png",np.greater_equal,evalue_thresholds=[1e-10, 1e-1, 1, 10],maxvalue = np.max(all_distances2))
pdb.set_trace()
#num_queries = len(ourhitsdict)
#num_targets = len(targets)
# denom = num_queries * num_targets
# plot_roc_curve(
#     distance_hits_dict,
#     all_hits_max,
#     True,
#     0,
#     numhits,
#     "ResNet1d/eval/individual_distances.png",
#     np.greater_equal,
#     evalue_thresholds=[1e-10, 1e-1, 1, 10],
# )
# plt.clf()
#plot_roc_curve(distance_sum_hits_dict,all_hits_max,True,0,numhits,"ResNet1d/eval/summed_distances.png",np.greater_equal,evalue_thresholds=[1e-10, 1e-1, 1, 10],maxvalue = np.max(all_distances2))
# ourhitsdict contains everything that phmmer max contains
plt.clf()

plt.scatter(all_scores, all_distances)
plt.xlabel("Hmmer Scores")
plt.ylabel("CNN Distances")
plt.savefig("individual-distances-scores.png")
plt.clf()
plt.scatter(all_scores2, all_distances2)
plt.xlabel("Hmmer Scores")
plt.ylabel("CNN Distances")
plt.savefig("summed-distances-scores.png")
pdb.set_trace()


hmmer_pairs = []
hmmer_max_pairs = []
our_pairs = []

for query, targetdict in all_hits.items():
    for target in targetdict.keys():
        hmmer_pairs.append(sorted([(query, target)]))

for query, targetdict in all_hits_max.items():
    for target in targetdict.keys():
        hmmer_max_pairs.append(sorted([(query, target)]))

for query, targetdict in ourhitsdict.items():
    for target in targetdict.keys():
        our_pairs.append(sorted([(query, target)]))

intersection = outersection = extras = []
for pair in our_pairs:
    if pair in hmmer_pairs:
        intersection.append(pair)
    elif pair in hmmer_max_pairs:
        outersection.append(pair)
    else:
        extras.append(pair)


pdb.set_trace()


num_overlap = sum([len(targets) for targets in intersection.values()])

print(
    f"{(len(prefilter_overlap)/(num_our_prefilter))*100}% of our hits are also in hmmers prefiltered results"
)
print()
print(
    f"{(len(outersection)/(num_our_prefilter))*100}% of our hits are NOT in hmmers prefiltered results"
)
print()
print(f"{(len(extras)/(num_our_prefilter))*100}% of our hits are not in any hmmer results")
print()
print(f"There are {len(missing)} HMMER hits that are not in our results.")

distances_in_intersection = {}
distances_in_outersection = {}
hmmer_data_intersection = {}
hmmer_data_outersection = {}

distances_in_extras = {}

# the ones that we filtered out, that they also filtered out
for query in prefilter_overlap:
    for target, distance in ourhitsdict[query].items():
        distances_in_intersection[target] = distance
    for target, data in all_hits[query].items():
        hmmer_data_intersection[target] = data

# the ones that we filtered out, that they did not filter
for query in outersection:
    for target, distance in ourhitsdict[query].items():
        distances_in_outersection[target] = distance
    for target, data in all_hits_max[query].items():
        hmmer_data_outersection[target] = data

for query in extras:
    for target, distance in ourhitsdict[query].items():
        distances_in_extras[target] = distance

import matplotlib.pyplot as plt

plt.hist(
    distances_in_intersection.values(),
    density=True,
    histtype="step",
    alpha=0.75,
    label="intersection",
)
plt.hist(
    distances_in_outersection.values(),
    density=True,
    histtype="step",
    alpha=0.75,
    label="outersection",
)
plt.hist(distances_in_extras.values(), density=True, histtype="step", alpha=0.75, label="other")

plt.legend()
plt.savefig("distances.png")
