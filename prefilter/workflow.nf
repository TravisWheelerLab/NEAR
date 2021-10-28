#!/usr/bin/env nextflow

afas = Channel.fromPath(params.afas)

process makedirs {

    script:
    """
    mkdir -p ${params.out_path_fasta}/${params.pid}
    """
}

process carbs_split {

    publishDir "${params.out_path_fasta}/${params.pid}"

    input:
        path afa from afas

    output:
       // TODO: is there a better way of doing the below?
       file '*train.fa*'  optional true into train_fasta
       file '*valid.fa*'  optional true into valid_fasta
       file '*test.fa*'  optional true into test_fasta

    """
    if [[ -f ${params.afa_directory}/${afa.baseName}.ddgm ]]
    then
        echo $afa
        carbs split -T argument --split_test --output_path . ${params.afa_directory}/${afa} ${params.pid}
    else
        n_seq=\$(grep ">" $params.afa_directory/${afa} | wc -l)
        if [[ \$n_seq > 1 ]]
        then
            carbs cluster $params.afa_directory/${afa} # run clustering if ddgm can't be found
            carbs split -T argument --split_test --output_path . ${params.afa_directory}/${afa} ${params.pid}
        else
        echo "only 1 sequence in ${afa}"
        fi
    fi
    """
}

process to_fasta_train {

    errorStrategy 'ignore'
    publishDir = "${params.out_path_fasta}/${params.pid}"

    input:
        file train from train_fasta

    output:
        stdout result_train

    script:
    """
    echo \$PATH
    domtblout_file=\$(echo ${train.baseName} | sed "s/${params.pid}-train/domtblout/g")
    domtblout_file=${params.domtblout_directory}/\$domtblout_file
    if [[ -f \$domtblout_file ]]
    then
        grep ">" $train | sed 's/>//g' | sed 's/ .*//g' | grep -f - \$domtblout_file | awk '{print \$1,\$4,\$5,\$7}' | label_fasta.py --fasta_file $train -
    else
        echo "couldn't find domtblout at \$domtblout_file $train"
    fi
    """
}

process to_fasta_test {

    errorStrategy 'ignore'
    publishDir = "${params.out_path_fasta}/${params.pid}"

    input:
        file test from test_fasta

    output:
        stdout result_test

    script:
    """
    domtblout_file=\$(echo ${test.baseName} | sed "s/${params.pid}-test/domtblout/g")
    domtblout_file=${params.domtblout_directory}/\$domtblout_file
    if [[ -f \$domtblout_file ]]
    then
        grep ">" $test | sed 's/>//g' | sed 's/ .*//g' | grep -f - \$domtblout_file | awk '{print \$1,\$4,\$5,\$7}' | label_fasta.py --fasta_file $test -
    else
        echo "couldn't find domtblout at \$domtblout_file $test"
    fi
    """
}

process to_fasta_valid {

    errorStrategy 'ignore'
    publishDir = "${params.out_path_fasta}"

    input:
        file valid from valid_fasta

    output:
        stdout result_valid

    script:
    """
    domtblout_file=\$(echo ${valid.baseName} | sed "s/${params.pid}-valid/domtblout/g")
    domtblout_file=${params.domtblout_directory}/\$domtblout_file
    if [[ -f \$domtblout_file ]]
    then
        grep ">" $valid | sed 's/>//g' | sed 's/ .*//g' | grep -f - \$domtblout_file | awk '{print \$1,\$4,\$5,\$7}' | label_fasta.py --fasta_file $valid -
    else
        echo "couldn't find domtblout at \$domtblout_file $valid"
    fi
    """
}
