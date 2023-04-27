"""Model comparison

You need to compare on data with query id 0 as well 

You need to increase max sequence length too"""

from src.data.benchmarking import (
    get_evaluation_data,
    plot_roc_curve,
    generate_roc_from_sorted_pairs,
)

import pdb


QUERY_FILENUM = 0
_, _, all_hits_max = get_evaluation_data(
    query_id=QUERY_FILENUM,
    targethitsfile="/xdisk/twheeler/daphnedemekas/ALL_PHMMERMAX_QUERY0_RESULTS.pkl",
)

_, _, all_hits_normal = get_evaluation_data(
    query_id=QUERY_FILENUM,
    targethitsfile="/xdisk/twheeler/daphnedemekas/ALL_PHMMERNORMAL_QUERY0_RESULST.pkl",
)

# TODO: need to parse normal for query id 4 as i want to compare these results

# QUERY_FILENUM=4


# querysequences, targetsequences, all_hits_max = get_evaluation_data(
#     query_id=QUERY_FILENUM, targethitsfile ='evaltargethmmerhits.pkl'
# )

# _, _, all_hits_normal = get_evaluation_data(
#     hitsdirpath = '/xdisk/twheeler/daphnedemekas/phmmer_normal_query_4_results',
#     query_id=QUERY_FILENUM,targethitsfile ='evaltargethmmerhitsnormal.pkl'
# )


BLOSUM_MODEL_RESULTS_PATH_FLAT = "/xdisk/twheeler/daphnedemekas/prefilter-output/BlosumEvaluation/similarities"

BLOSUM_MODEL_RESULTS_PATH_IVF = "/xdisk/twheeler/daphnedemekas/prefilter-output/BlosumEvaluation/similarities-IVF"

ALIGNMENT_MODEL_RESULTS_PATH_IVF = "/xdisk/twheeler/daphnedemekas/prefilter-output/AlignmentEvaluation/similarities-IVF"

ALIGNMENT_MODEL_RESULTS_PATH_IVF_0 = "/xdisk/twheeler/daphnedemekas/prefilter-output/AlignmentEvaluation/similarities-IVF-0"

# align data
TEMP_DATA_FILE = "/xdisk/twheeler/daphnedemekas/data1_distance_sum_hits.txt"
TEMP_DATA_FILE_NORMAL = (
    "/xdisk/twheeler/daphnedemekas/data1_distance_sum_hits_normal.txt"
)

TEMP_DATA_FILE_0 = (
    "/xdisk/twheeler/daphnedemekas/data1_distance_sum_hits_0.txt"
)
TEMP_DATA_FILE_NORMAL_0 = (
    "/xdisk/twheeler/daphnedemekas/data1_distance_sum_hits_normal_0.txt"
)

# blosum data
TEMP_DATA_FILE = (
    "/xdisk/twheeler/daphnedemekas/blosum_data1_distance_sum_hits.txt"
)
TEMP_DATA_FILE_NORMAL = (
    "/xdisk/twheeler/daphnedemekas/blosum_data1_distance_sum_hits_normal.txt"
)

# generate_roc_from_sorted_pairs(BLOSUM_MODEL_RESULTS_PATH_IVF, "/xdisk/twheeler/daphnedemekas/sorted_blosum_IVF_pairs.pkl", TEMP_DATA_FILE, all_hits_max, "ResNet1d/eval/blosum_IVF_roc.png")
# generate_roc_from_sorted_pairs(BLOSUM_MODEL_RESULTS_PATH_IVF, "/xdisk/twheeler/daphnedemekas/sorted_blosum_IVF_pairs_normal.pkl", TEMP_DATA_FILE_NORMAL, all_hits_normal, "ResNet1d/eval/blosum_IVF_roc_normal.png")


ALIGNMENT_MODEL_RESULTS_PATH_FLAT = "/xdisk/twheeler/daphnedemekas/prefilter-output/AlignmentEvaluation/similarities"

TEMP_DATA_FILE_ALIGN_FLAT = (
    "/xdisk/twheeler/daphnedemekas/blosum_data1_distance_align_flat.txt"
)
# generate_roc_from_sorted_pairs(ALIGNMENT_MODEL_RESULTS_PATH_FLAT, "/xdisk/twheeler/daphnedemekas/sorted_alignment_flat_pairs_0.pkl", TEMP_DATA_FILE_ALIGN_FLAT, all_hits_max, "ResNet1d/eval/alignment_flat_roc_0.png")
# generate_roc_from_sorted_pairs(ALIGNMENT_MODEL_RESULTS_PATH_IVF_0, "/xdisk/twheeler/daphnedemekas/sorted_alignment_IVF_pairs_0_normal.pkl", TEMP_DATA_FILE_NORMAL_0, all_hits_normal, "ResNet1d/eval/alignment_IVF_roc_0_normal.png")
# NEED TO REDO THIS LAST ONE -- YOU HAD IT AS NORMAL
generate_roc_from_sorted_pairs(
    ALIGNMENT_MODEL_RESULTS_PATH_IVF_0,
    "/xdisk/twheeler/daphnedemekas/sorted_alignment_IVF_pairs_0.pkl",
    TEMP_DATA_FILE_0,
    all_hits_max,
    "ResNet1d/eval/alignment_IVF_roc_0.png",
)

# generate_roc_from_sorted_pairs("/xdisk/twheeler/daphnedemekas/sorted_alignment_IVF_pairs.pkl", TEMP_DATA_FILE, all_hits_normal, "ResNet1d/eval/alignment_IVF_roc_normal.png")
# plot_roc_curve("ResNet1d/eval/alignment_IVF_roc_normal.png", [13331, 30072, 32812, 33245], 1308720793, filename = TEMP_DATA_FILE_NORMAL)
# plot_roc_curve("ResNet1d/eval/alignment_IVF_roc.png", [11454, 24468, 33649, 84215], 1341468330, filename = TEMP_DATA_FILE)
