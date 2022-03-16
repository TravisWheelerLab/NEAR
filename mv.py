from glob import glob
from shutil import copy
import os
import prefilter
import prefilter.utils as utils
import yaml


f = "/Users/mac/Dropbox/notebook/prefilter/2022-03-14/PF02171_seed.afa"
l, s = utils.msa_from_file(f)
print(l)
