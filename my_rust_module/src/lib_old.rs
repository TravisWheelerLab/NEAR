use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::collections::HashSet; // Import HashSet

//#[pyfunction]
// fn filter_scores(scores_array_list: Vec<Vec<Vec<Vec<f64>>>>,
//     indices_array_list: Vec<Vec<Vec<Vec<usize>>>>,
//     unrolled_names: Vec<String>) -> Vec<HashMap<String, f64>> {
//     filter_scores_in_parallel(scores_array_list, indices_array_list, unrolled_names)
// }

// fn filter_scores_in_parallel(scores_array_list: Vec<Vec<Vec<Vec<f64>>>>,
//     indices_array_list: Vec<Vec<Vec<Vec<usize>>>>,
//     unrolled_names: Vec<String>) -> Vec<HashMap<String, f64>> {
//     scores_array_list.par_iter().zip(indices_array_list.par_iter()).flat_map(|(scores_array, indices_array)| {_filter(&scores_array, &indices_array, &unrolled_names)}).collect()
//     }
fn read_scores(filename: &str) -> io::Result<Vec<Vec<Vec<f64>>>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);

    let mut matrix = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let parts: Vec<_> = line.split(" : ").collect();
        let row: Vec<f64> = parts[0].split(", ").filter_map(|s| s.parse().ok()).collect();
        let column: Vec<f64> = parts[1].split(", ").filter_map(|s| s.parse().ok()).collect();

        if row.len() != column.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Mismatched row and column lengths"));
        }

        let combined: Vec<Vec<f64>> = row.into_iter().zip(column.into_iter()).map(|(r,c)| vec![r,c]).collect();
        matrix.push(combined);
    }

    Ok(matrix)
}

fn read_indices(filename: &str) -> io::Result<Vec<Vec<Vec<usize>>>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);

    let mut matrix = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let parts: Vec<_> = line.split(" : ").collect();

        let row: Vec<usize> = parts[0].split(", ")
                                     .filter_map(|s| s.parse().ok())
                                     .collect();

        let column: Vec<usize> = parts[1].split(", ")
                                        .filter_map(|s| s.parse().ok())
                                        .collect();

        if row.len() != column.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Mismatched row and column lengths"));
        }

        let combined: Vec<Vec<usize>> = row.into_iter().zip(column.into_iter()).map(|(r,c)| vec![r,c]).collect();
        matrix.push(combined);
    }

    Ok(matrix)
}


fn read_names(filename: &str) -> io::Result<Vec<String>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);

    let names: Vec<_> = reader.lines().collect::<Result<_, _>>()?;

    Ok(names)
}

#[pyfunction]
fn filter_scores(
) -> Vec<HashMap<String, f64>> {


    let scores_array_list = read_scores("/xdisk/twheeler/daphnedemekas/all_scores.txt")?;
    let indices_array_list = read_indices("/xdisk/twheeler/daphnedemekas/all_indices.txt")?;
    let unrolled_names = read_names("/xdisk/twheeler/daphnedemekas/unrolled_names.txt")?;


    let chunk_size = scores_array_list.len() / 16;
    let scores_chunks: Vec<_> = scores_array_list.chunks(chunk_size).collect();
    let indices_chunks: Vec<_> = indices_array_list.chunks(chunk_size).collect();

    scores_chunks
        .par_iter()
        .zip(indices_chunks.par_iter())
        .map(|(scores_chunk, indices_chunk)| {
            _filter(
                scores_chunk.to_vec(),
                indices_chunk.to_vec(),
                &unrolled_names,
            )
        })
        .flatten()
        .collect()
}

fn _filter(
    scores_array_list: Vec<Vec<Vec<f64>>>,
    indices_array_list: Vec<Vec<Vec<usize>>>,
    unrolled_names: &Vec<String>,
) -> Vec<HashMap<String, f64>> {
    //init();
    //scores_array_list.par_iter().zip(indices_array_list.par_iter()).map(|(scores_array, indices_array)| {
    let mut filtered_scores_list = Vec::new();
    for (scores_array, indices_array) in scores_array_list.iter().zip(indices_array_list.iter()) {
        let mut filtered_scores: HashMap<String, f64> = HashMap::new();

        for match_idx in 0..scores_array.len() {
            let match_scores = &scores_array[match_idx];
            let indices = &indices_array[match_idx];
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

            let sorted_names: Vec<_> = sorted_match_idx
                .iter()
                .map(|&idx| names[idx].clone())
                .collect();
            let sorted_indices: Vec<_> = sorted_match_idx.iter().map(|&idx| indices[idx]).collect();
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

    filtered_scores_list
    //}
    //        filtered_scores

    //   })
    // .collect()
}

#[pymodule]
fn my_rust_module(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(filter_scores, m)?)?;
    Ok(())
}
