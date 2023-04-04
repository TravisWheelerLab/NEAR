
all_hits_max_file_4 = '/xdisk/twheeler/daphnedemekas/ALL_PHMMERMAX_QUERY4_RESULTS.pkl.pkl'
all_hits_normal_file_4 = '/xdisk/twheeler/daphnedemekas/ALL_PHMMERNORMAL_QUERY4_RESULTS.pkl.pkl'

all_hits_max_file_0 = '/xdisk/twheeler/daphnedemekas/ALL_PHMMERMAX_QUERY0_RESULTS.pkl.pkl'
all_hits_normal_file_0 = '/xdisk/twheeler/daphnedemekas/ALL_PHMMERNORMAL_QUERY0_RESULTS.pkl.pkl'


ALIGNMENT_MODEL_RESULTS_PATH_IVF = (
    "/xdisk/twheeler/daphnedemekas/prefilter-output/AlignmentEvaluation/similarities-IVF"
)

ALIGNMENT_MODEL_RESULTS_PATH_IVF_0 = (
    "/xdisk/twheeler/daphnedemekas/prefilter-output/AlignmentEvaluation/similarities-IVF-0"
)

ALIGNMENT_MODEL_RESULTS_PATH_FLAT = (
    "/xdisk/twheeler/daphnedemekas/prefilter-output/AlignmentEvaluation/similarities"
)

BLOSUM_MODEL_RESULTS_PATH_IVF = (
    "/xdisk/twheeler/daphnedemekas/prefilter-output/BlosumEvaluation/similarities-IVF"
)

BLOSUM_MODEL_RESULTS_PATH_FLAT = (
    "/xdisk/twheeler/daphnedemekas/prefilter-output/BlosumEvaluation/similarities"
)
ALIGNMENT_MODEL_RESULTS_PATH_SCANN = (
    "/xdisk/twheeler/daphnedemekas/prefilter-output/AlignmentEvaluation/similarities-scann"
)
KMER_MODEL_RESULTS_PATH = "/xdisk/twheeler/daphnedemekas/prefilter-output/AlignmentEvaluation/similarities-kmer"

ALIGNMENT_MODEL_IVF_0_MAX_DATAFILE = "/xdisk/twheeler/daphnedemekas/temp_files/align_roc_data_ivf_max_0.txt"

ALIGNMENT_MODEL_IVF_4_MAX_DATAFILE = "/xdisk/twheeler/daphnedemekas/temp_files/align_roc_data_ivf_max_4.txt"

ALIGNMENT_MODEL_IVF_0_NORMAL_DATAFILE = "/xdisk/twheeler/daphnedemekas/temp_files/align_roc_data_ivf_normal_0.txt"

ALIGNMENT_MODEL_IVF_4_NORMAL_DATAFILE = "/xdisk/twheeler/daphnedemekas/temp_files/align_roc_data_ivf_normal_4.txt"

ALIGNMENT_MODEL_FLAT_DATAFILE =  "/xdisk/twheeler/daphnedemekas/temp_files/align_roc_data_flat.txt"

ALIGNMENT_MODEL_SCANN_DATAFILE =  "/xdisk/twheeler/daphnedemekas/temp_files/align_scann.txt"

BLOSUM_MODEL_IVF_MAX_DATAFILE = "/xdisk/twheeler/daphnedemekas/temp_files/blosum_roc_data_ivf_max.txt"

BLOSUM_MODEL_IVF_NORMAL_DATAFILE  = "/xdisk/twheeler/daphnedemekas/temp_files/blosum_roc_data_ivf_normal.txt"

KMER_MODEL_DATAFILE  = "/xdisk/twheeler/daphnedemekas/temp_files/kmer_data.txt"
KMER_MODEL_DATAFILE_NORMAL  = "/xdisk/twheeler/daphnedemekas/temp_files/kmer_data_normal.txt"

def load_alignment_inputs(hits, mode):
    if mode == "max":
        return {"model_results_path" : ALIGNMENT_MODEL_RESULTS_PATH_IVF, "hmmer_hits_dict": hits, "data_savedir": '/xdisk/twheeler/daphnedemekas/alignment_model_ivf_4', 
                                                "evaluemeansfile": "evaluemeans_align_ivf_4", "evaluemeanstitle": "Correlation in ALIGN IVF model - HMMER Max", "sorted_alignment_pairs_path": "/xdisk/twheeler/daphnedemekas/sorted_alignment_ivf_pairs_4.pkl",
                                              "temp_data_file": ALIGNMENT_MODEL_IVF_4_MAX_DATAFILE, "roc_filepath": "ResNet1d/eval/align_ivf_roc_4.png", "num_pos_per_evalue": [482667, 1105431, 1519838, 3722920], "num_hits":1341468330,  "plot_roc" : False}
    elif mode == "normal":
        return {"model_results_path" : ALIGNMENT_MODEL_RESULTS_PATH_IVF, "hmmer_hits_dict": hits, "data_savedir": '/xdisk/twheeler/daphnedemekas/alignment_model_ivf_4_normal', 
                                            "evaluemeansfile": "evaluemeans_align_ivf_normal", "evaluemeanstitle": "Correlation in ALIGN IVF model - HMMER Normal", "sorted_alignment_pairs_path": "/xdisk/twheeler/daphnedemekas/sorted_alignment_ivf_pairs_4_normal.pkl",
                                            "temp_data_file": ALIGNMENT_MODEL_IVF_4_NORMAL_DATAFILE, "roc_filepath": "ResNet1d/eval/align_ivf_roc_normal_4.png","num_pos_per_evalue": [482384, 1038554, 1081989, 1088067], "num_hits":1341468330,  "plot_roc" : False}
    elif mode == "flat":
        return {"model_results_path" : ALIGNMENT_MODEL_RESULTS_PATH_FLAT, "hmmer_hits_dict": hits, "data_savedir": '/xdisk/twheeler/daphnedemekas/alignment_model_flat', 
                                            "evaluemeansfile": "evaluemeans_align_flat", "evaluemeanstitle": "Correlation in ALIGN Flat model - HMMER Max", "sorted_alignment_pairs_path": "/xdisk/twheeler/daphnedemekas/sorted_alignment_flat_pairs_0.pkl",
                                            "temp_data_file": ALIGNMENT_MODEL_FLAT_DATAFILE, "roc_filepath": "ResNet1d/eval/align_flat_roc.png", "num_pos_per_evalue": [459492, 1049796, 1444675, 3544063], "num_hits": 893168453,  "plot_roc" : False}
    
    elif mode == "scann":
        return {"model_results_path" : ALIGNMENT_MODEL_RESULTS_PATH_SCANN, "hmmer_hits_dict": hits, "data_savedir": '/xdisk/twheeler/daphnedemekas/alignment_model_scann', 
                                            "evaluemeansfile": "evaluemeans_align_scann", "evaluemeanstitle": "Correlation in ALIGN SCANN model - HMMER Max", "sorted_alignment_pairs_path": "/xdisk/twheeler/daphnedemekas/sorted_alignment_scann_pairs.pkl",
                                            "temp_data_file": ALIGNMENT_MODEL_SCANN_DATAFILE, "roc_filepath": "ResNet1d/eval/align_scann_roc.png", "plot_roc" : True}

    
    else: raise Exception("mode not understood")


def load_blosum_inputs(hits, mode):
    if mode == "max":
        return {"model_results_path" : BLOSUM_MODEL_RESULTS_PATH_IVF, "hmmer_hits_dict": hits, "data_savedir": '/xdisk/twheeler/daphnedemekas/blosum_model_ivf', 
                                            "evaluemeansfile": "evaluemeans_blosum_ivf", "evaluemeanstitle": "Correlation in BLOSUM IVF model - HMMER Max", "sorted_alignment_pairs_path": "/xdisk/twheeler/daphnedemekas/sorted_blosum_IVF_pairs.pkl",
                                            "temp_data_file": BLOSUM_MODEL_IVF_MAX_DATAFILE, "roc_filepath": "ResNet1d/eval/blosum_IVF_roc.png", "num_pos_per_evalue": [458272, 997128, 1329747, 3085143], "num_hits": 1217740527,
                                            "plot_roc":False}
    elif mode == "normal":
        return {"model_results_path" : BLOSUM_MODEL_RESULTS_PATH_IVF, "hmmer_hits_dict": hits, "data_savedir": '/xdisk/twheeler/daphnedemekas/blosum_model_ivf_normal', 
                                            "evaluemeansfile": "evaluemeans_blosum_ivf_normal", "evaluemeanstitle": "Correlation in BLOSUM IVF model - HMMER Normal", "sorted_alignment_pairs_path": "/xdisk/twheeler/daphnedemekas/sorted_blosum_IVF_pairs_normal.pkl",
                                            "temp_data_file": BLOSUM_MODEL_IVF_NORMAL_DATAFILE, "roc_filepath": "ResNet1d/eval/blosum_IVF_roc_normal.png","num_pos_per_evalue": [458019, 940240, 974535, 979231], "num_hits": 1217740527, "plot_roc":False}
    elif mode == "flat":
        return {"model_results_path" : BLOSUM_MODEL_RESULTS_PATH_FLAT, "hmmer_hits_dict": hits, "data_savedir": '/xdisk/twheeler/daphnedemekas/blosum_model_flat', 
                                            "evaluemeansfile": "evaluemeans_blosum_flat", "evaluemeanstitle": "Correlation in BLOSUM Flat model - HMMER Max", "sorted_alignment_pairs_path": "/xdisk/twheeler/daphnedemekas/sorted_blosum_flat_pairs.pkl",
                                            "temp_data_file": "/xdisk/twheeler/daphnedemekas/blosum_roc_data_flat_max.txt", "roc_filepath": "ResNet1d/eval/blosum_flat_roc.png", "num_pos_per_evalue": [260490, 593803, 808761, 1917479], "num_hits": 449863226, "plot_roc":False}

    else: raise Exception("mode not understood")

def load_kmer_inputs(hits, mode):
    if mode == "max":
        return {"model_results_path" : KMER_MODEL_RESULTS_PATH, "hmmer_hits_dict": hits, "data_savedir": '/xdisk/twheeler/daphnedemekas/kmer_model_max', 
                                            "evaluemeansfile": "evaluemeans_align_kmer_max", "evaluemeanstitle": "Correlation in ALIGN Kmer model - HMMER Max", "sorted_alignment_pairs_path": "/xdisk/twheeler/daphnedemekas/sorted_alignment_kmer_pairs_max.pkl",
                                            "temp_data_file": KMER_MODEL_DATAFILE, "roc_filepath": "ResNet1d/eval/align_kmer_roc.png"}
    elif mode == "normal":
        return {"model_results_path" : KMER_MODEL_RESULTS_PATH, "hmmer_hits_dict": hits, "data_savedir": '/xdisk/twheeler/daphnedemekas/kmer_model_normal', 
                                            "evaluemeansfile": "evaluemeans_align_kmer_normal", "evaluemeanstitle": "Correlation in ALIGN Kmer model - HMMER Normal", "sorted_alignment_pairs_path": "/xdisk/twheeler/daphnedemekas/sorted_alignment_kmer_pairs_normal.pkl",
                                            "temp_data_file": KMER_MODEL_DATAFILE_NORMAL, "roc_filepath": "ResNet1d/eval/align_kmer_roc_normal.png"}
            
    else: raise Exception("mode not understood")
