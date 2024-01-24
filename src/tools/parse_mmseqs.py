import os
import pickle
import tqdm


def parse_mmseqs(outputdir, mmseqspath):
    # assert len(queries) == 16769
    # print(f"Number of relevant targets: {len(mytargets)}")
    # outputdir = "/xdisk/twheeler/daphnedemekas/prefilter-output/mmseqs-reversed"

    # for subdir in tqdm.tqdm(os.listdir(directory)):
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    # mmseqspath = f"/xdisk/twheeler/daphnedemekas/mmseqs-rev/alnRes.m8"
    with open(mmseqspath, "r") as f:
        lines = f.readlines()
        for f in tqdm.tqdm(lines):
            line_split = f.split()
            query, target = line_split[0], line_split[1]
            #        if query not in myqueries or target not in mytargets:
            #            print("skipping")
            #            continue
            bitscore = float(line_split[-1])
            if os.path.exists(f"{outputdir}/{query}.txt"):
                with open(f"{outputdir}/{query}.txt", "a") as f:
                    f.write(f"{target}     {bitscore}" + "\n")
            else:
                with open(f"{outputdir}/{query}.txt", "w") as f:
                    f.write(f"{target}     {bitscore}" + "\n")


def parse_mmseqs_prefilter(
    outputdir, mmseqspath, target_lookup_path, query_lookup_path
):
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    target_ids = {}
    with open(target_lookup_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            linesplit = line.split()
            id = linesplit[0]
            name = linesplit[1]
            if id == "125932":
                print(id)
                print(name)
            target_ids[id] = name
    query_ids = {}
    with open(query_lookup_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            linesplit = line.split()
            id = linesplit[0]
            name = linesplit[1]
            query_ids[id] = name
    print(target_ids["125932"]) 
    #print(list(target_ids.keys())[:-10])
    print(f"Length of target ids: {len(target_ids)}")
    print(f"Length of query ids :{len(query_ids)}")
    for i in range(28):
        path = mmseqspath + f".{i}"
        print(f"Processing {path}")
        with open(path, "r") as f:
            lines = f.readlines()
            for f in tqdm.tqdm(lines):
                line_split = f.split()
                try:
                    target_id, query_id, score = line_split[0], line_split[1], line_split[2]
                except:
                    print("Skipping line")
                    continue
                #print('0' in target_ids)
                #print(str(target_id) in target_ids)
                target_id = str(target_id.strip()).replace('\x00', '')
                query_id = str(query_id.strip())
                #if target_id not in target_ids:
                    #if target_id == "125932":
                    #    print("yes")
                    #print(f"Target id {target_id} not in target ids")
                #if query_id not in query_ids:
                #    print(f"Query id {query_id} not in query ids")
                #assert target_id in target_ids, f"Target id {target_id} not in target ids"
                #assert query_id in query_ids, "Query id not in query ids"

                query = query_ids[query_id]
                target = target_ids[target_id]

                if os.path.exists(f"{outputdir}/{query}.txt"):
                    with open(f"{outputdir}/{query}.txt", "a") as f:
                        f.write(f"{target}     {score}" + "\n")
                else:
                    with open(f"{outputdir}/{query}.txt", "w") as f:
                        f.write(f"{target}     {score}" + "\n")


if __name__ == "__main__":
    outputdir = "/xdisk/twheeler/daphnedemekas/prefilter-output/mmseqs-prefilter"

    mmseqspath = f"/xdisk/twheeler/daphnedemekas/mmseqs-prefilter"

    target_lookup_path = "/xdisk/twheeler/daphnedemekas/mmseqs_target_DB.lookup"
    query_lookup_path = "/xdisk/twheeler/daphnedemekas/mmseqs_query_DB.lookup"

    parse_mmseqs_prefilter(outputdir, mmseqspath, target_lookup_path, query_lookup_path)

    print("Done parsing forward mmseqs")

    outputdir = (
        "/xdisk/twheeler/daphnedemekas/prefilter-output/mmseqs-prefilter-reversed"
    )

    mmseqspath = f"/xdisk/twheeler/daphnedemekas/mmseqs-prefilter-reversed"

    target_lookup_path = "/xdisk/twheeler/daphnedemekas/mmseqs_target_DB_reversed.lookup"
    query_lookup_path = "/xdisk/twheeler/daphnedemekas/mmseqs_query_DB.lookup"

    parse_mmseqs_prefilter(outputdir, mmseqspath, target_lookup_path, query_lookup_path)
    print("Done parsing reversed mmseqs")
