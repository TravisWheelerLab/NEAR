from label_fasta import parse_domtblout

df = parse_domtblout("/home/tc229954/subset/120_Rick_ant.0.5-boogle.domtblout")
print(df.head())
