from sacred import Experiment

ex = Experiment()

@ex.config
def config():
    gpus = 4
    num_nodes = 1
    num_workers = 0
    log_dir = "models/may27/18_layer_resnet_with_attn_no_pos_enc"
    batch_size = 16
    epochs = 20
    learning_rate = 1e-4
    seq_len = 400
    apply_attention = True
    msa_transformer = False
