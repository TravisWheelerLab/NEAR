use rayon::prelude::*;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::env;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs::File;
use std::io::Write;
use std::io::{self, BufRead, BufReader};
use std::time::Instant;
//use ndarray::Array2;
extern crate hdf5;
use ndarray::Array2;
thread_local! {
    static MATCH_SCORES: RefCell<Vec<f64>> = RefCell::new(vec![0.0; 1000]);
    static INDICES: RefCell<Vec<usize>> = RefCell::new(vec![0; 1000]);
    static NAMES: RefCell<Vec<&'static str>> = RefCell::new(vec![""; 1000]);
    static SORTED_MATCH_IDX: RefCell<Vec<usize>> = RefCell::new((0..1000).collect());
    static SORTED_NAMES: RefCell<Vec<&'static str>> = RefCell::new(vec![""; 1000]);
    static SORTED_INDICES: RefCell<Vec<usize>> = RefCell::new(vec![0; 1000]);
    static SORTED_MATCHES: RefCell<Vec<f64>> = RefCell::new(vec![0.0; 1000]);
}

fn read_hdf5_to_vec_f64(path: &str) -> Result<Vec<Vec<Vec<f64>>>, hdf5::Error> {
    let file = hdf5::File::open(path)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
        .expect("Failed to open HDF5 file");
    let mut all_data: Vec<Vec<Vec<f64>>> = Vec::new();

    let mut i = 0;
    loop {
        match file.dataset(&format!("array_{}", i)) {
            Ok(dataset) => {
                let data: Array2<f64> = dataset.read_2d()?;
                let mut row_vecs = Vec::new();
                for row in data.outer_iter() {
                    row_vecs.push(row.to_vec());
                }
                all_data.push(row_vecs);
                //all_data.push(data.into_raw_vec());
            }
            Err(_) => break,
        }
        i += 1;
    }

    Ok(all_data)
}

fn read_hdf5_to_vec_usize(path: &str) -> Result<Vec<Vec<Vec<usize>>>, hdf5::Error> {
    let file = hdf5::File::open(path)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
        .expect("Failed to open HDF5 file");
    let mut all_data: Vec<Vec<Vec<usize>>> = Vec::new();

    let mut i = 0;
    loop {
        match file.dataset(&format!("array_{}", i)) {
            Ok(dataset) => {
                let data: Array2<usize> = dataset.read_2d()?;
                //               all_data.push(data.into_raw_vec());
                let mut row_vecs = Vec::new();
                for row in data.outer_iter() {
                    row_vecs.push(row.to_vec());
                }
                all_data.push(row_vecs);
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

fn filter_scores_inner() -> Vec<HashMap<String, f64>>{
    // Assuming you have your scores_array_list and indices_array_list defined above here
    //let mut filtered_scores_list = Vec::new();

    println!("In new rust module");
    let scores_array_list =
        read_hdf5_to_vec_f64("/xdisk/twheeler/daphnedemekas/all-scores-250-rev.h5")
            .expect("Failed to read HDF5 data");
    let indices_array_list =
        read_hdf5_to_vec_usize("/xdisk/twheeler/daphnedemekas/all-indices-250-rev.h5")
            .expect("Failed to read HDF5 data");
    let target_names = read_names("/xdisk/twheeler/daphnedemekas/unrolled-names-reversed.txt")
        .expect("Failed to read unrolled names");
    println!("The length of scores array is: {}", scores_array_list.len());
    println!(
        "The length of scores array is: {}",
        indices_array_list.len()
    );
    println!("The length of target names: {}", target_names.len());

    let filtered_scores_list: Vec<_> = scores_array_list
        .par_iter()
        .zip(indices_array_list.par_iter())
        .map(|(scores_array, indices_array)| {
            let mut filtered_scores: HashMap<String, f64> = HashMap::new();

            MATCH_SCORES.with(|match_scores_ref| {
                INDICES.with(|indices_ref| {
                    NAMES.with(|names_ref| {
                        SORTED_MATCH_IDX.with(|sorted_match_idx_ref| {
                            SORTED_NAMES.with(|sorted_names_ref| {
                                SORTED_INDICES.with(|sorted_indices_ref| {
                                    SORTED_MATCHES.with(|sorted_matches_ref| {
                                        for match_idx in 0..scores_array.len() {
                                            let match_scores = &scores_array[match_idx];
                                            let indices = &indices_array[match_idx];

                                            let names: Vec<_> = indices
                                                .iter()
                                                .map(|&idx| target_names[idx].clone())
                                                .collect();

                                            let mut sorted_match_idx =
                                                sorted_match_idx_ref.borrow_mut();
                                            sorted_match_idx.sort_unstable_by(|&a, &b| {
                                                match match_scores[b].partial_cmp(&match_scores[a])
                                                {
                                                    Some(ordering) => ordering,
                                                    None => Ordering::Equal,
                                                }
                                            });

                                            let sorted_names: Vec<_> = sorted_match_idx
                                                .iter()
                                                .filter_map(|&idx| names.get(idx))
                                                .collect();
                                            let sorted_indices: Vec<_> = sorted_match_idx
                                                .iter()
                                                .filter_map(|&idx| indices.get(idx))
                                                .collect();
                                            let sorted_matches: Vec<_> = sorted_match_idx
                                                .iter()
                                                .map(|&idx| match_scores[idx])
                                                .collect();

                                            let mut unique_values = HashSet::new();
                                            let mut unique_indices = Vec::new();

                                            for (index, &value) in sorted_names.iter().enumerate() {
                                                if unique_values.insert(value) {
                                                    unique_indices.push(index);
                                                }
                                            }

                                            let new_indices: Vec<_> = unique_indices
                                                .iter()
                                                .map(|&idx| sorted_indices[idx])
                                                .collect();
                                            let new_names: Vec<_> = new_indices
                                                .iter()
                                                .map(|&idx| target_names[*idx].clone())
                                                .collect();
                                            let new_scores: Vec<_> = unique_indices
                                                .iter()
                                                .map(|&idx| sorted_matches[idx])
                                                .collect();

                                            for (distance, name) in
                                                new_scores.iter().zip(new_names.iter())
                                            {
                                                *filtered_scores
                                                    .entry(name.to_string())
                                                    .or_insert(0.0) += *distance;
                                            }
                                        }
                                    });
                                });
                            });
                        });
                    });
                });
            });

            filtered_scores
        })
        .collect();
 //   filtered_scores_inner
    println!("The length of filtered scores is: {}", filtered_scores_list.len()); 
    return filtered_scores_list
 // Now use or return filtered_scores_list as needed
}
fn write(
    filtered_scores_list: &Vec<HashMap<String, f64>>,
    output_path: &str,
    query_names: &Vec<String>,
) {
    filtered_scores_list.par_iter().enumerate().for_each(|(i, filtered_scores)| {
        let file_name = format!("{}/{}.txt", output_path, query_names[i]);
        let mut f = File::create(&file_name).expect("Unable to create file");

        f.write_all(b"Name     Distance\n")
            .expect("Unable to write data");

        for (name, distance) in filtered_scores.iter() {
            let line = format!("{}     {}\n", name, distance);
            f.write_all(line.as_bytes()).expect("Unable to write data");
        }
    });
}

fn time_it<F, T>(func: F) -> T
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let result = func();
    let duration = start.elapsed();
    println!("Function executed in: {:?}", duration);
    result
}

fn main() {
    time_it(|| {
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
        let filtered_scores = filter_scores_inner();        
//match filter_scores_inner() {
 //           Ok(filtered_scores) => {
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
    });
}

