from src.datasets.datasets import AlignmentGenerator


train_dataset_args = {
    "ali_path": "/xdisk/twheeler/daphnedemekas/stk_alignments/0/**/",
    "seq_len": 256,
}

generator = AlignmentGenerator(**train_dataset_args)
