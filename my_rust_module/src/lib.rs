use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::collections::HashMap;
use std::collections::HashSet; // Import HashSet
use std::cmp::Ordering;
use rayon::prelude::*;

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
    

#[pyfunction]
fn filter_scores(scores_array_list: &Vec<Vec<Vec<f64>>>,
    indices_array_list: &Vec<Vec<Vec<usize>>>,
    unrolled_names: &Vec<String>) -> Vec<HashMap<String, f64> {
        scores_array_list.chunks(96).collect::<Vec<_>>()
        .par_iter()
        .flat_map(|chunk| {chunk.iter().zip(indices_array_list.iter()).flat_map(|(scores_array, indices_array)| {
            _filter(&scores_array, &indices_array, unrolled_names)
        })
})
.collect::<Vec<_>>()
}


fn _filter(
    scores_array_list: &Vec<Vec<Vec<f64>>>,
    indices_array_list: &Vec<Vec<Vec<usize>>>,
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
            let names: Vec<_> = indices.iter().map(|&idx| unrolled_names[idx].clone()).collect();
            

            let mut sorted_match_idx: Vec<usize> = (0..match_scores.len()).collect();
            sorted_match_idx.sort_unstable_by(|&a, &b| {
                match match_scores[b].partial_cmp(&match_scores[a]) {
                    Some(ordering) => ordering,
                    None => Ordering::Equal,
                }
            });
            
            let sorted_names: Vec<_> = sorted_match_idx.iter().map(|&idx| names[idx].clone()).collect();
            let sorted_indices: Vec<_> = sorted_match_idx.iter().map(|&idx| indices[idx]).collect();
            let sorted_matches: Vec<_> = sorted_match_idx.iter().map(|&idx| match_scores[idx]).collect();


            // Create a HashSet to store the unique values
            let mut unique_values = HashSet::new();
            let mut unique_indices = Vec::new();

            // Iterate over the elements of some_array along with their indices
            for (index, &value) in sorted_names.iter().enumerate() {
                if unique_values.insert(value) {
                // If the value is not already in the HashSet, add it to unique_indices
                unique_indices.push(index);
                }
            }

            let new_indices: Vec<_> = unique_indices.iter().map(|&idx| sorted_indices[idx]).collect();
            let new_names: Vec<_> = new_indices.iter().map(|&idx| unrolled_names[idx].clone()).collect();
            let new_scores: Vec<_> = unique_indices.iter().map(|&idx| sorted_matches[idx]).collect();

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

