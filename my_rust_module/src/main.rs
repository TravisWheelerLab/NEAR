//use pyo3::prelude::*;
//use pyo3::wrap_pyfunction;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::collections::HashSet; // Import HashSet
use std::env;
use std::fs::File;
use std::io::Write;
use std::io::{self, BufRead, BufReader};
extern crate hdf5;

fn read_hdf5_to_vec_f64(path: &str) -> hdf5::Result<Vec<Vec<Vec<f64>>>> {
    let file = hdf5::File::open(path)?;
    let mut all_data: Vec<Vec<Vec<f64>>> = Vec::new();

    let i = 0;
    loop {
        match file.dataset(&format!("array_{}", i)) {
            Ok(dataset) => {
                let data: hdf5::Array2<f64> = dataset.read_2d()?;
                all_data.push(data.to_vec());
            }
            Err(_) => break,
        }
        i += 1;
    }

    Ok(all_data)
}

fn read_hdf5_to_vec_usize(path: &str) -> hdf5::Result<Vec<Vec<Vec<usize>>>> {
    let file = hdf5::File::open(path)?;
    let mut all_data: Vec<Vec<Vec<usize>>> = Vec::new();

    let i = 0;
    loop {
        match file.dataset(&format!("array_{}", i)) {
            Ok(dataset) => {
                let data: hdf5::Array2<usize> = dataset.read_2d()?;
                all_data.push(data.to_vec());
            }
            Err(_) => break,
        }
        i += 1;
    }

    Ok(all_data)
}

fn read_names(filename: &str) -> io::Result<Vec<String>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);

    let names: Vec<_> = reader.lines().collect::<Result<_, _>>()?;

    Ok(names)
}

fn filter_scores_inner() -> Result<Vec<HashMap<String, f64>>, std::io::Error> {
    let mut filtered_scores_list = Vec::new();

    println!("In new rust module");
    let scores_array_list = read_hdf5_to_vec_f64("/xdisk/twheeler/daphnedemekas/all_scores.h5")?;
    let indices_array_list =
        read_hdf5_to_vec_usize("/xdisk/twheeler/daphnedemekas/all_indices.h5")?;
    let unrolled_names = read_names("/xdisk/twheeler/daphnedemekas/unrolled_names-reversed.txt")?;

    println!("The length of scores array is: {}", scores_array_list.len());
    println!(
        "The length of scores array is: {}",
        indices_array_list.len()
    );
    println!("The length of unrolled names: {}", unrolled_names.len());
    let mut idx = 0;
    for (scores_array, indices_array) in scores_array_list.iter().zip(indices_array_list.iter()) {
        idx += 1;
        println!("{idx}");
        let mut filtered_scores: HashMap<String, f64> = HashMap::new();

        for match_idx in 0..scores_array.len() {
            let match_scores = &scores_array[match_idx];
            let indices = &indices_array[match_idx];
            //println!("match_idx {}", match_idx);
            //println!("indices {:?}", indices);
            let names: Vec<_> = indices
                .iter()
                .map(|&idx| unrolled_names[idx].clone())
                .collect();

            let mut sorted_match_idx: Vec<usize> = (0..match_scores.len()).collect();
            sorted_match_idx.sort_unstable_by(|&a, &b| {
                match match_scores[b].partial_cmp(&match_scores[a]) {
                    Some(ordering) => ordering,
                    None => Ordering::Equal,
                }
            });

            //let sorted_names: Vec<_> = sorted_match_idx.iter().map(|&idx| names[idx].clone()).collect();
            //let sorted_indices: Vec<_> = sorted_match_idx.iter().map(|&idx| indices[idx]).collect();
            //let sorted_matches: Vec<_> = sorted_match_idx.iter().map(|&idx| match_scores[idx]).collect();
            let sorted_names: Vec<_> = sorted_match_idx
                .iter()
                .map(|&idx| names[idx].clone())
                .collect();
            //let sorted_indices: Vec<_> = sorted_match_idx.iter().filter_map(|&idx| indices.get(idx)).collect();

            let sorted_indices: Vec<usize> =
                sorted_match_idx.iter().map(|&idx| indices[idx]).collect();
            let sorted_matches: Vec<_> = sorted_match_idx
                .iter()
                .map(|&idx| match_scores[idx])
                .collect();

            // Create a HashSet to store the unique values
            let mut unique_values = HashSet::new();
            let mut unique_indices = Vec::new();

            // Iterate over the elements of some_array along with their indices
            for (index, &ref value) in sorted_names.iter().enumerate() {
                if unique_values.insert(value) {
                    // If the value is not already in the HashSet, add it to unique_indices
                    unique_indices.push(index);
                }
            }

            let new_indices: Vec<_> = unique_indices
                .iter()
                .map(|&idx| sorted_indices[idx])
                .collect();
            let new_names: Vec<_> = new_indices
                .iter()
                .map(|&idx| unrolled_names[idx].clone())
                .collect();
            let new_scores: Vec<_> = unique_indices
                .iter()
                .map(|&idx| sorted_matches[idx])
                .collect();

            //println!("unique indices {:?}", unique_indices);
            //println!("sorted_names {:?}", sorted_names);
            for (distance, name) in new_scores.iter().zip(new_names.iter()) {
                *filtered_scores.entry(name.to_string()).or_insert(0.0) += *distance;
            }
        }

        filtered_scores_list.push(filtered_scores);
    }

    Ok(filtered_scores_list)
}

fn write(
    filtered_scores_list: &Vec<HashMap<String, f64>>,
    output_path: &str,
    query_names: &Vec<String>,
) {
    for (i, filtered_scores) in filtered_scores_list.iter().enumerate() {
        let file_name = format!("{}/{}.txt", output_path, query_names[i]);
        let mut f = File::create(&file_name).expect("Unable to create file");

        f.write_all(b"Name     Distance\n")
            .expect("Unable to write data");

        for (name, distance) in filtered_scores.iter() {
            let line = format!("{}     {}\n", name, distance);
            f.write_all(line.as_bytes()).expect("Unable to write data");
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    // Ensure there are enough arguments provided
    if args.len() != 4 {
        eprintln!(
            "Usage: {} <query_filename> <output_path> <write_results>",
            args[0]
        );
        return;
    }
    let query_filename = &args[1];
    let output_path = &args[2];
    let write_results: bool = match args[3].parse() {
        Ok(val) => val,
        Err(_) => {
            eprintln!("Error: write_results must be a boolean (true or false)");
            return;
        }
    };
    match filter_scores_inner() {
        Ok(filtered_scores) => {
            if write_results {
                match read_names(query_filename) {
                    Ok(query_names) => {
                        write(&filtered_scores, output_path, &query_names);
                    }
                    Err(e) => {
                        eprintln!("Error reading names: {}", e);
                        return;
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("Error filtering scores: {}", e);
            return;
        }
    }
}
