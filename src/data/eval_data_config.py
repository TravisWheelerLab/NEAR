all_hits_max_file_4 = "data/evaltargetdictmax"
all_hits_normal_file_4 = "data/evaltargetdictnormal"
all_hits_max_file_4 = "data/evaluationtargetdict"
all_hits_normal_file_4 = "data/evaluationtargetdictnormal"
# all_hits_max_file_4 = "data/evaluationtargetdict"
# all_hits_normal_file_4 = "data/evaluationtargetdictnormal"


def load_inputs(hits, mode, modelname):
    if mode == "max":
        return {
            "hmmer_hits_dict": hits,
            "temp_file": f"/xdisk/twheeler/daphnedemekas/temp_files/{modelname}_max",
        }  # "num_pos_per_evalue": [482667, 1105431, 1519838, 3722920], "num_hits":1341468330,  "plot_roc" : False}
    elif mode == "normal":
        return {
            "hmmer_hits_dict": hits,
            "temp_file": f"/xdisk/twheeler/daphnedemekas/temp_files/{modelname}_normal",
        }  # "num_pos_per_evalue": [482384, 1038554, 1081989, 1088067], "num_hits":1341468330,  "plot_roc" : False}
