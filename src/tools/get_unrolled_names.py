
target_names_file = "target_names.txt"
target_lengths_file = "target_lengths.txt"
unrolled_names_file = '/xdisk/twheeler/daphnedemekas/unrolled_names.txt'

def unroll(target_names_file, target_lengths_file, unrolled_names_file):
    target_names = open(target_names_file, "r")
    target_lengths = open(target_lengths_file, "r")
    unrolled_names = []
    for name, length in zip(target_names.readlines(), target_lengths.readlines()):
        unrolled_names.extend([name.strip("\n")] * int(length.strip("\n")))

    with open(unrolled_names_file, "w") as f:
        for name in unrolled_names:
            f.write(name + "\n")


unroll(target_names_file, target_lengths_file, unrolled_names_file)



target_names_file = "reversed-target-names.txt"
target_lengths_file = "reversed-target-lengths.txt"
unrolled_names_file = '/xdisk/twheeler/daphnedemekas/unrolled_names_reversed.txt'

unroll(target_names_file, target_lengths_file, unrolled_names_file)



