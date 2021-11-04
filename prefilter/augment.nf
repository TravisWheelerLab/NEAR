#!/usr/bin/env nextflow

split_fasta = channel.fromPath(params.fasta_path)

process fetch_alignment {

    publishDir "${params.out_path_fasta}/${params.pid}"

    input:
        file fasta from split_fasta

    output:
       path "*augment" into output

    script:
    """
    n_seq=\$(grep ">" ${fasta} | wc -l)
    if [[ \$n_seq <= ${params.augment_threshold} ]]
    then
        name=\$(echo ${fasta.baseName} | sed 's/.${params.pid}-train//g')
        esl-reformat stockholm ${params.afa_directory}/\$name.afa > \$name-train.sto
        grep ">" $fasta | sed 's/>//g' | sed 's/ .*//g' > names
        esl-alimanip --seq-k names \$name-train.sto
        hmmbuild \$name-train.hmm \$name-train.sto
        hmmemit -N ${params.n_emissions} -o ${fasta}.augment \$name-train.hmm
    fi
    """
}
