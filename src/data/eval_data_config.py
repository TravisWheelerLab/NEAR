all_hits_max_file_4 = "data/evaluationtargetdict"
all_hits_normal_file_4 = "data/evaluationtargetdictnormal"


def load_knn_inputs(hits, mode, modelname):
    if mode == "max":
        return {
            "model_results_path": f"/xdisk/twheeler/daphnedemekas/prefilter-output/{modelname}",
            "hmmer_hits_dict": hits,
            "data_savedir": f"/xdisk/twheeler/daphnedemekas/{modelname}",
            "evaluemeansfile": f"evaluemeans_{modelname}_max",
            "evaluemeanstitle": f"Correlation in {modelname} model - HMMER Max",
            "roc_filepath": f"ResNet1d/results/{modelname}_roc_max.png",
            "plot_roc": True,
            "temp_file": f"/xdisk/twheeler/daphnedemekas/temp_files/{modelname}",
        }  # "num_pos_per_evalue": [482667, 1105431, 1519838, 3722920], "num_hits":1341468330,  "plot_roc" : False}
    elif mode == "normal":
        return {
            "model_results_path": f"/xdisk/twheeler/daphnedemekas/prefilter-output/{modelname}",
            "hmmer_hits_dict": hits,
            "data_savedir": f"/xdisk/twheeler/daphnedemekas/{modelname}normal",
            "evaluemeansfile": f"evaluemeans_{modelname}_normal",
            "evaluemeanstitle": f"Correlation in {modelname} model - HMMER Normal",
            "roc_filepath": f"ResNet1d/results/{modelname}_roc_normal.png",
            "plot_roc": True,
            "temp_file": f"/xdisk/twheeler/daphnedemekas/temp_files/{modelname}normal",
        }  # "num_pos_per_evalue": [482384, 1038554, 1081989, 1088067], "num_hits":1341468330,  "plot_roc" : False}

def load_mmseqs_inputs(hits, mode, modelname):
    if mode == "max":
        return {
            "model_results_path": f"/xdisk/twheeler/daphnedemekas/prefilter-output/{modelname}",
            "hmmer_hits_dict": hits,
            "data_savedir": f"/xdisk/twheeler/daphnedemekas/{modelname}",
            "evaluemeansfile": f"evaluemeans_{modelname}_max",
            "evaluemeanstitle": f"Correlation in {modelname} model - HMMER Max",
            "roc_filepath": f"ResNet1d/results/{modelname}_roc_max.png",
            "plot_roc": True,
            "temp_file": f"/xdisk/twheeler/daphnedemekas/temp_files/{modelname}",
        }  # "num_pos_per_evalue": [482667, 1105431, 1519838, 3722920], "num_hits":1341468330,  "plot_roc" : False}
    elif mode == "normal":
        return {
            "model_results_path": f"/xdisk/twheeler/daphnedemekas/prefilter-output/{modelname}",
            "hmmer_hits_dict": hits,
            "data_savedir": f"/xdisk/twheeler/daphnedemekas/{modelname}normal",
            "evaluemeansfile": f"evaluemeans_{modelname}_normal",
            "evaluemeanstitle": f"Correlation in {modelname} model - HMMER Normal",
            "roc_filepath": f"ResNet1d/results/{modelname}_roc_normal.png",
            "plot_roc": True,
            "temp_file": f"/xdisk/twheeler/daphnedemekas/temp_files/{modelname}normal",
        }  # "num_pos_per_evalue": [482384, 1038554, 1081989, 1088067], "num_hits":1341468330,  "plot_roc" : False}

def load_esm_inputs(hits, mode, modelname):
    if mode == "max":
        return {
            "model_results_path": f"/xdisk/twheeler/daphnedemekas/prefilter-output/{modelname}",
            "hmmer_hits_dict": hits,
            "data_savedir": f"/xdisk/twheeler/daphnedemekas/{modelname}",
            "evaluemeansfile": f"evaluemeans_{modelname}_max",
            "evaluemeanstitle": f"Correlation in {modelname} model - HMMER Max",
            "roc_filepath": f"ResNet1d/results/{modelname}_roc_max.png",
            "plot_roc": False,
            "temp_file": f"/xdisk/twheeler/daphnedemekas/temp_files/{modelname}",
        }  # "num_pos_per_evalue": [482667, 1105431, 1519838, 3722920], "num_hits":1341468330,  "plot_roc" : False}
    elif mode == "normal":
        return {
            "model_results_path": f"/xdisk/twheeler/daphnedemekas/prefilter-output/{modelname}",
            "hmmer_hits_dict": hits,
            "data_savedir": f"/xdisk/twheeler/daphnedemekas/{modelname}normal",
            "evaluemeansfile": f"evaluemeans_{modelname}_normal",
            "evaluemeanstitle": f"Correlation in {modelname} model - HMMER Normal",
            "roc_filepath": f"ResNet1d/results/{modelname}_roc_normal.png",
            "plot_roc": False,
            "temp_file": f"/xdisk/twheeler/daphnedemekas/temp_files/{modelname}normal",
        }  # "num_pos_per_evalue": [482384, 1038554, 1081989, 1088067], "num_hits":1341468330,  "plot_roc" : False}


def load_alignment_inputs(hits, mode, modelname):
    if mode == "max":
        return {
            "model_results_path": "/xdisk/twheeler/daphnedemekas/prefilter-output/AlignmentEvaluation/"
            + modelname,
            "hmmer_hits_dict": hits,
            "data_savedir": f"/xdisk/twheeler/daphnedemekas/{modelname}_max",
            "evaluemeansfile": f"evaluemeans_align_{modelname}_max",
            "evaluemeanstitle": f"Correlation in ALIGN IVF model - HMMER Max",
            "roc_filepath": f"ResNet1d/results/{modelname}_max_roc.png",
            "plot_roc": True,
            "temp_file": f"/xdisk/twheeler/daphnedemekas/temp_files/{modelname}_max",
        }
    elif mode == "normal":
        return {
            "model_results_path": "/xdisk/twheeler/daphnedemekas/prefilter-output/AlignmentEvaluation/"
            + modelname,
            "hmmer_hits_dict": hits,
            "data_savedir": f"/xdisk/twheeler/daphnedemekas/{modelname}_normal",
            "evaluemeansfile": f"evaluemeans_align_{modelname}_normal",
            "evaluemeanstitle": f"Correlation in ALIGN IVF model - HMMER Normal",
            "roc_filepath": f"ResNet1d/results/{modelname}_normal_roc.png",
            "plot_roc": True,
            "temp_file": f"/xdisk/twheeler/daphnedemekas/temp_files/{modelname}_normal",
        }

    else:
        raise Exception("mode not understood")
