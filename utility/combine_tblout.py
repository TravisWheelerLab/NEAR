import sys
import re
import os



if len(sys.argv) != 3:
    print("Usage: combine_tblout.py input_dir output_csv")
    exit(1)

input_dir = sys.argv[1]
output_file = sys.argv[2]


out_file = open(output_file, 'w')

for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(".tblout"):
                full_path = os.path.join(root, filename)
                with open(full_path, 'r') as file:
                    for line in file:
                        # Skip comments and empty lines
                        if line.startswith("#") or not line.strip():
                            continue

                        # Split the line into columns
                        columns = re.split(r'\s+', line.strip())

                        # Extract the desired fields
                        try:
                            target_name = columns[0]
                            query_name = columns[2]
                            seq_e_value = columns[4]
                            seq_score = columns[5]
                        except IndexError:
                            print(f"Skipping malformed line: {line.strip()}")
                        
                        out_file.write(query_name + " " + target_name + " " + seq_e_value + "\n")


out_file.close()