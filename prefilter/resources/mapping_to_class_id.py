import pandas as pd
import json

pfam_name_file = "./pfam_accession_ids.txt"

with open(pfam_name_file, "r") as src:
    names = src.read().split("\n")

pfam_accession_id_to_class_code = {}
class_code = 0
for accession_id in names:
    if accession_id not in pfam_accession_id_to_class_code and len(accession_id) > 0:
        pfam_accession_id_to_class_code[accession_id] = class_code
        class_code += 1

with open("accession_id_to_class_code.json", "w") as dst:
    json.dump(pfam_accession_id_to_class_code, dst)
