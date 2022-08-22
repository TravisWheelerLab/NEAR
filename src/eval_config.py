from sacred import Experiment

evaluation_ex = Experiment()


def functor(seq):
    return seq


@evaluation_ex.config
def config():

    gpus = 1
    model_name = "ResNet1d"
    evaluator_name = "UniRefEvaluator"
    model_path = "model_data/aug22/single_epoch_run/ResNet1d/1/"

    evaluator_args = {
        "query_file": "/home/u4/colligan/data/prefilter/uniref_benchmark/Q_benchmark2k30k.fa",
        "target_file": "/home/u4/colligan/data/prefilter/uniref_benchmark/T_benchmark2k30k.fa",
        "encoding_func": functor,
    }
