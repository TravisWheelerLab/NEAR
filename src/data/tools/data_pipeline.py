from .cluster_target_data import main as cluster 


targetfastasdir ='/xdisk/twheeler/daphnedemekas/prefilter/uniref/split_subset/targets/'

traintargetspath= '/xdisk/twheeler/daphnedemekas/targetdataseqs/train-final.txt'
evaltargetspath= '/xdisk/twheeler/daphnedemekas/targetdataseqs/eval-final.txt'
targetclusterdir= '/xdisk/twheeler/daphnedemekas/targets_clustered-final'


# trainalignmentspath= "/xdisk/twheeler/daphnedemekas/train-alignments"
# evalalignmentspath= "/xdisk/twheeler/daphnedemekas/eval-alignments"

cluster(traintargetspath, evaltargetspath, targetclusterdir, targetfastasdir)

