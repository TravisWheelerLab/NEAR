#!/usr/bin/env nextflow


params.pid = 0.5
params.afa = "/Users/mac/data/prefilter/testing/*afa"
params.out_path_fasta = "/Users/mac/data/prefilter/testing/fasta/${params.pid}"
params.out_path_json = "/Users/mac/data/prefilter/testing/json/${params.pid}"
params.domtblout_directory = "/Users/mac/data/prefilter/testing/"
params.afa_directory = "/Users/mac/data/prefilter/testing/"
params.evalue_threshold = 1e-5

afas = Channel.fromPath(params.afa)

process makedirs {
    
    output:
	stdout result

    """
    mkdir -p ${params.out_path_fasta}
    mkdir -p ${params.out_path_json}
    """
}

process carbs_split {
    storeDir "${params.out_path_fasta}"

    input:
        path afa from afas

    output:
       file "*.fa" into split_fasta_files
       path afa into afa

    """
    carbs split -T argument --split_test --output_path ${params.out_path_fasta} ${params.afa_directory}/${afa} ${params.pid}
    """
}

process to_json {

    storeDir = "${params.out_path_json}"

    // the line below iterates over split_fasta_files
    input:
	each file(fasta) from split_fasta_files.flatMap()
	path afa from afa

    output:
        file "*.json" into json_files

    """
    bash $PROJECT_WORK_DIR/convert_domtblout_to_json.sh ${fasta} ${params.domtblout_directory}/${afa.baseName}.domtblout ${params.out_path_json} ${params.evalue_threshold}
    """
}
