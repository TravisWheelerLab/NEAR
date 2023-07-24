
import os 
import tqdm
msv_output = '/xdisk/twheeler/daphnedemekas/msv-output/output.txt'

#with open(msv_output, 'r') as f:
#    lines = f.readlines()

#firstquery = lines[10]

#print(firstquery)
#query = firstquery.split()[1]
#queryfile = open(f"/xdisk/twheeler/daphnedemekas/prefilter-output/msv-reversed/{query}.txt", "w")



file1 = open(msv_output, 'r')
count = 0
 
while True:

    count += 1
    if count < 9:
        continue
    if count == 11:
        print(f"First query : {line}")
        query = line.split()[1]
        queryfile = open(f"/xdisk/twheeler/daphnedemekas/prefilter-output/msv/{query}.txt", "w") 
        continue
    # Get next line from file
    line = file1.readline() 
    # if line is empty
    # end of file is reached
    if not line:
        break

    if "Query:" in line:
        queryfile.close()
        query = line.split()[1]
        print(f"Query: {query}")
        queryfile = open(f"/xdisk/twheeler/daphnedemekas/prefilter-output/msv/{query}.txt", "w")
    elif "MSV filter and score: " in line:
        linesplit = line.split()
        try:
            target = linesplit[5]
        except:
            print(linesplit)
            raise
        score = 1-float(linesplit[6])

        queryfile.write(f"{target}     {score}" + "\n")


