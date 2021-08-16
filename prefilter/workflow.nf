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
       path afa into train_afa
       path afa into valid_afa
       path afa into test_afa

    """
    if [[ -f ${params.afa_directory}/${afa.baseName}.ddgm ]]
    then
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
        path afa from train_afa

    script:
    """
    if [[ -f ${params.domtblout_directory}/${afa.baseName}.domtblout ]]
    then
        bash convert_domtblout_to_json.sh ${train} ${params.domtblout_directory}/${afa.baseName}.domtblout ${params.out_path_json}/${params.pid} ${params.evalue_threshold}
    else
        echo "run hmmsearch on your un-clustered sequences!"
        exit 1
    fi
    """
    // TODO: figure out why nextflow can't see the json files on output if i add
    // output:
    //    *-train.json* into train_json
}

process to_json_test {

    publishDir = "${params.out_path_json}/${params.pid}"

    input:
        file test from test_fasta
        path afa from test_afa

    script:
    """
    bash convert_domtblout_to_json.sh ${test} ${params.domtblout_directory}/${afa.baseName}.domtblout ${params.out_path_json}/${params.pid} ${params.evalue_threshold}
    """
}

process to_json_valid {

    publishDir = "${params.out_path_json}"

    input:
        file valid from valid_fasta
        path afa from valid_afa

    script:
    """
    bash convert_domtblout_to_json.sh ${valid} ${params.domtblout_directory}/${afa.baseName}.domtblout ${params.out_path_json}/${params.pid} ${params.evalue_threshold}
    """
}