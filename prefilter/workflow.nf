#!/usr/bin/env nextflow

afas = Channel.fromPath(params.afas)

process makedirs {

    script:
    """
    mkdir -p ${params.out_path_fasta}/${params.pid}
    mkdir -p ${params.out_path_json}/${params.pid}
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

process to_json_train {

    publishDir = "${params.out_path_json}/${params.pid}"

    input:
        file train from train_fasta

    output:
        stdout result_train

    script:
    """
    domtblout_file=\$(echo ${train.baseName} | sed "s/${params.pid}-train/domtblout/g")
    domtblout_file=${params.domtblout_directory}/\$domtblout_file
    if [[ -f \$domtblout_file ]]
    then
        bash convert_domtblout_to_json.sh ${train} \$domtblout_file ${params.out_path_json}/${params.pid} ${params.evalue_threshold}
    else
        echo "couldn't find domtblout at \$domtblout_file $train"
    fi
    """
}

process to_json_test {

    publishDir = "${params.out_path_json}/${params.pid}"

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
        bash convert_domtblout_to_json.sh ${test} \$domtblout_file ${params.out_path_json}/${params.pid} ${params.evalue_threshold}
    else
        echo "couldn't find domtblout at \$domtblout_file"
    fi
    """
}

process to_json_valid {

    publishDir = "${params.out_path_json}"

    input:
        file valid from valid_fasta

    output:
        stdout result_valid

    script:
    // if a valid file exists, it's always in the format {}.$pid-valid.fa. All I need to do is replace that with .domtblout.
    """
    domtblout_file=\$(echo ${valid.baseName} | sed "s/${params.pid}-valid/domtblout/g")
    domtblout_file=${params.domtblout_directory}/\$domtblout_file
    if [[ -f \$domtblout_file ]]
    then
        bash convert_domtblout_to_json.sh ${valid} \$domtblout_file ${params.out_path_json}/${params.pid} ${params.evalue_threshold}
    else
        echo "couldn't find domtblout at \$domtblout_file"
    fi
    """
}
// result_test.view{ it.trim() }
// result_train.view{ it.trim() }
// result_valid.view{ it.trim() }