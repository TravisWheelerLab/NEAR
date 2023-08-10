use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::collections::HashMap;
use std::collections::HashSet; // Import HashSet
use std::cmp::Ordering;
extern crate hdf5;
use ndarray::Array2;
use std::fs::File;
use std::io::Write;
use std::io::{self, BufRead, BufReader};


fn read_hdf5_to_vec_f64(path: &str) -> Result<Vec<Vec<Vec<f64>>>, hdf5::Error> {
    let file = hdf5::File::open(path).map_err(|e| io::Error::new(io::ErrorKind::Other, e)).expect("Failed to open HDF5 file");
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
    let file = hdf5::File::open(path).map_err(|e| io::Error::new(io::ErrorKind::Other, e)).expect("Failed to open HDF5 file");
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

#[pyfunction]
fn filter_scores(
    scores_array_list: Vec<Vec<Vec<f64>>>,
    indices_array_list: Vec<Vec<Vec<usize>>>,
    target_names: Vec<String>,
) -> Vec<HashMap<String, f64>> {
//) -> Result<Vec<HashMap<String, f64>>, std::io::Error> {
    let mut filtered_scores_list = Vec::new();

    println!("In new rust module");
    //let scores_array_list = read_scores("/xdisk/twheeler/daphnedemekas/all_scores-reversed.txt")?;
    //let indices_array_list = read_indices("/xdisk/twheeler/daphnedemekas/all_indices-reversed.txt")?;
    //let unrolled_names = read_names("/xdisk/twheeler/daphnedemekas/unrolled_names.txt")?;
    let scores_array_list = read_hdf5_to_vec_f64("/xdisk/twheeler/daphnedemekas/all_scores_test.h5").expect("Failed to read HDF5 data");
    let indices_array_list =
        read_hdf5_to_vec_usize("/xdisk/twheeler/daphnedemekas/all_indices_test.h5").expect("Failed to read HDF5 data");

    let mut idx = 0;
    let mut names: Vec<&str> = vec![&""; 1000];

    //let match_indices: Vec<usize> =  (0..1000).collect();

    let mut sorted_match_idx: Vec<usize> = (0..1000).collect();

    let mut sorted_names: Vec<&str> = vec![&""; 1000];

    let mut sorted_indices: Vec<usize> = vec![0; 1000];

    let mut sorted_matches: Vec<f64> = vec![0.0; 1000];

    for (scores_array, indices_array) in scores_array_list.iter().zip(indices_array_list.iter()) {
        idx += 1;
        println!("{idx}");
        let mut filtered_scores: HashMap<String, f64> = HashMap::new();

        for match_idx in 0..scores_array.len() {
            let match_scores = &scores_array[match_idx];

            let indices = &indices_array[match_idx];

            for (i, &idx) in indices.iter().enumerate() {
                names[i] = &target_names[idx];
            }

            sorted_match_idx.sort_unstable();

            sorted_match_idx.sort_unstable_by(|&a, &b| {
                match match_scores[b].partial_cmp(&match_scores[a]) {
                    Some(ordering) => ordering,
                    None => Ordering::Equal,
                }
            });

            //unsafe {
            let mut i = 0;
            for &idx in &sorted_match_idx {
                sorted_names[i] = &names[idx];
                sorted_indices[i] = indices[idx];
                sorted_matches[i] = match_scores[idx];
                i += 1;
            }
            //}
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
                .map(|&idx| target_names[idx].clone())
                .collect();
            let new_scores: Vec<_> = unique_indices
                .iter()
                .map(|&idx| sorted_matches[idx])
                .collect();

          for (distance, name) in new_scores.iter().zip(new_names.iter()) {
                *filtered_scores.entry(name.to_string()).or_insert(0.0) += *distance;
            }
        }

        filtered_scores_list.push(filtered_scores);
    }

    filtered_scores_list
}

#[pymodule]
fn my_rust_module(_py: Python, m: &PyModule) -> PyResult<()> {
 
    //#[pyfn(m, "filter_scores")]
   // fn filter(py: Python, scores_array_list: Vec<Vec<Vec<f64>>>,indices_array_list: Vec<Vec<Vec<usize>>>,unrolled_names: Vec<String>) -> PyResult<Vec<HashMap<String, f64>>> {
  //      let filtered_scores = py.allow_threads(move || filter_scores(scores_array_list, indices_array_list, unrolled_names));    
 //       Ok(filtered_scores)
//    }
    // Add your Rust functions to the Python module
    m.add_function(wrap_pyfunction!(filter_scores, m)?)?;
    Ok(())
}
