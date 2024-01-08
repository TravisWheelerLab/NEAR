def filter(datafile_reversed):
    filtered_targets_file = open(
        "/xdisk/twheeler/daphnedemekas/prefilter/data/filtered_target_names.txt", "r"
    )
    filtered_targets = [
        f.strip().strip("\n") for f in filtered_targets_file.readlines()
    ]
    file = open(datafile_reversed, "r")
    newfile = open(datafile_reversed, "w")

    for line in file:
        target = line.split("          ")[1]

        if target in filtered_targets:
            newfile.write(line)


filter("/xdisk/twheeler/daphnedemekas/temp_files/esm_max")
