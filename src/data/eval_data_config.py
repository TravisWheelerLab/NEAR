import pickle

# all_hits_max_file_4 = "data/evaltargetdictmax" THIS WONT WORK
# all_hits_normal_file_4 = "data/evaltargetdictnormal"
all_hits_max_file_4 = "data/evaluationtargetdict"
all_hits_normal_file_4 = "data/evaluationtargetdictnormal"

# all_hits_max_file_4 = "data/hmmerhits-masked-dict"


def load_inputs(hits, modelname, norm_q=True, norm_t=True):
    if "msv" in modelname:
        roc_filepath = f"ResNet1d/results/{modelname}_roc.png"
        temp_file = f"/xdisk/twheeler/daphnedemekas/temp_files/{modelname}"
    else:
        if norm_q and not norm_t:
            roc_filepath = f"ResNet1d/results/{modelname}_roc_norm_q.png"
            temp_file = f"/xdisk/twheeler/daphnedemekas/temp_files/{modelname}_norm_q"
        elif norm_t and not norm_q:
            roc_filepath = f"ResNet1d/results/{modelname}_roc_norm_t.png"
            temp_file = f"/xdisk/twheeler/daphnedemekas/temp_files/{modelname}_norm_t"

        elif norm_t and norm_q:
            roc_filepath = f"ResNet1d/results/{modelname}_roc_normalised.png"
            temp_file = (
                f"/xdisk/twheeler/daphnedemekas/temp_files/{modelname}_normalised"
            )

        else:
            roc_filepath = f"ResNet1d/results/{modelname}_roc.png"
            temp_file = f"/xdisk/twheeler/daphnedemekas/temp_files/{modelname}"
        if "masked" in all_hits_max_file_4 and "CPU" in modelname:
            roc_filepath = roc_filepath[:-4] + "-masked.png"
            temp_file = temp_file + "-masked"
    print(f"Roc filepath: {roc_filepath}")
    print(f"temp file: {temp_file}")
    if "masked" in modelname:
        # TODO different hmmer hits for masked

        return {
            "model_results_path": f"/xdisk/twheeler/daphnedemekas/prefilter-output/{modelname}",
            "hmmer_hits_dict": hits,
            "data_savedir": f"/xdisk/twheeler/daphnedemekas/{modelname}",
            "evaluemeansfile": f"evaluemeans_{modelname}",
            "evaluemeanstitle": f"Correlation in {modelname} model ",
            "roc_filepath": roc_filepath,
            "plot_roc": True,
            "temp_file": temp_file,  # f"/xdisk/twheeler/daphnedemekas/temp_files/{modelname}_masked",
            "query_lengths_file": "data/query-lengths-masked.pkl",
            "target_lengths_file": "data/target-lengths-masked.pkl",
            "norm_q": norm_q,
            "norm_t": norm_t,
        }  # "num_pos_per_evalue": [482667, 1105431, 1519838, 3722920], "num_hits":1341468330,  "plot_roc" : False}
    else:
        return {
            "model_results_path": f"/xdisk/twheeler/daphnedemekas/prefilter-output/{modelname}",
            "hmmer_hits_dict": hits,
            "data_savedir": f"/xdisk/twheeler/daphnedemekas/{modelname}",
            "evaluemeansfile": f"evaluemeans_{modelname}",
            "evaluemeanstitle": f"Correlation in {modelname} model",
            "roc_filepath": roc_filepath,
            "plot_roc": True,
            "temp_file": temp_file,  # f"/xdisk/twheeler/daphnedemekas/temp_files/{modelname}",
            "query_lengths_file": "data/query-lengths.pkl",
            "target_lengths_file": "data/target-lengths.pkl",
            "norm_q": norm_q,
            "norm_t": norm_t,
        }
