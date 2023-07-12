import os 

train_paths = '/xdisk/twheeler/daphnedemekas/train-alignments-final'
eval_paths = '/xdisk/twheeler/daphnedemekas/eval-alignments-final'

for path in os.listdir(train_paths):
    if not os.path.exists(path.strip("\n")):
        