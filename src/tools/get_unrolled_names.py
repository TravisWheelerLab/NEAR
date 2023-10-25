target_names = open("/xdisk/twheeler/daphnedemekas/reversed-target-names.txt", "r")
target_lengths = open("/xdisk/twheeler/daphnedemekas/reversed-target-lengths.txt", "r")
unrolled_names = []
for name, length in zip(target_names.readlines(), target_lengths.readlines()):
    unrolled_names.extend([name].strip("\n") * int(length.strip("\n")))

with open("/xdisk/twheeler/daphnedemekas/unrolled-names-reversed.txt", "w") as f:
    for name in unrolled_names:
        f.write(name + "\n")
