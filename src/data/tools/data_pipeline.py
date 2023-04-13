from .cluster_target_data import main as cluster 


targetfastasdir ='/xdisk/twheeler/daphnedemekas/prefilter/uniref/split_subset/targets/'

traintargetspath= '/xdisk/twheeler/daphnedemekas/targetdataseqs/train-95.txt'
evaltargetspath= '/xdisk/twheeler/daphnedemekas/targetdataseqs/eval-95.txt'
targetclusterdir= '/xdisk/twheeler/daphnedemekas/targets_clustered-95'


# trainalignmentspath= "/xdisk/twheeler/daphnedemekas/train-alignments"
# evalalignmentspath= "/xdisk/twheeler/daphnedemekas/eval-alignments"

cluster(traintargetspath, evaltargetspath, targetclusterdir, targetfastasdir)

