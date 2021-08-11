#!/usr/bin/env nextflow

TARGET_DIR = "1k"

params.pid = 0.34
params.afa = "$HOME/data/prefilter/$TARGET_DIR/*.afa"
params.out_path_fasta = "$HOME/data/prefilter/fasta/${params.pid}"
params.out_path_json = "$HOME/data/prefilter/json/${params.pid}"
params.domtblout_directory = "$HOME/data/prefilter/$TARGET_DIR/"
params.afa_directory = "$HOME/data/prefilter/$TARGET_DIR/"
params.evalue_threshold = 1e-5

params.filter = 'NO_FILE'

afas = Channel.fromPath(params.afa)

process makedirs {

    """
    mkdir -p ${params.out_path_fasta}
    mkdir -p ${params.out_path_json}
    """
}

process carbs_split {

    publishDir "${params.out_path_fasta}"

    input:
        path afa from afas

    output:
       file '*train.fa*'  optional true into train_fasta
       file '*valid.fa*'  optional true into valid_fasta
       file '*test.fa*'  optional true into test_fasta
       path afa into train_afa
       path afa into valid_afa
       path afa into test_afa

    // The below only runs once for each file in afas...
    """
    if [[ -f $params.afa_directory/${afa.baseName}.ddgm ]]; then
        carbs split -T argument --split_test --output_path . ${params.afa_directory}/${afa} ${params.pid}
    else
        carbs cluster $params.afa_directory/${afa} # run clustering if ddgm can't be found
        carbs split -T argument --split_test --output_path . ${params.afa_directory}/${afa} ${params.pid}
    fi
    """
}

process to_json_train {

    storeDir = "${params.out_path_json}"

    // the line below iterates over split_fasta_files
    input:
        file train from train_fasta
        // file test from test_fasta
        path afa from train_afa

    output:
        file "*.json" into json_train
    """
    if [[ -f ${params.domtblout_directory}/${afa.baseName}.domtblout ]]; then
        bash convert_domtblout_to_json.sh ${train} ${params.domtblout_directory}/${afa.baseName}.domtblout ${params.out_path_json} ${params.evalue_threshold}
    else
        echo "run hmmsearch on your un-clustered sequences!"
        exit 1
    fi
    """
}

process to_json_test {

    storeDir = "${params.out_path_json}"

    // the line below iterates over split_fasta_files
    input:
        file test from test_fasta
        path afa from test_afa

    output:
        file "*.json" into json_test

    """
    bash convert_domtblout_to_json.sh ${test} ${params.domtblout_directory}/${afa.baseName}.domtblout ${params.out_path_json} ${params.evalue_threshold}
    """
}

process to_json_valid {

    storeDir = "${params.out_path_json}"

    // the line below iterates over split_fasta_files
    input:
        file valid from valid_fasta
        path afa from valid_afa

    output:
        file "*.json" into json_valid

    """
    bash convert_domtblout_to_json.sh ${valid} ${params.domtblout_directory}/${afa.baseName}.domtblout ${params.out_path_json} ${params.evalue_threshold}
    """
}