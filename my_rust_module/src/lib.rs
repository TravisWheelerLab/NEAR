use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;
use std::collections::HashMap;

#[pyfunction]
fn filter_scores(
    scores_list: Vec<Vec<Vec<f64>>>,
    indices_list: Vec<Vec<Vec<usize>>>,
    unrolled_names: Vec<String>,
) -> PyResult<Vec<Py<PyDict>>> {
    // Call the existing filter_scores_impl function for each query
    let mut filtered_scores_list = Vec::new();
    for i in 0..scores_list.len() {
        let filtered_scores =
            filter_scores_impl(&scores_list[i], &indices_list[i], &unrolled_names);
        let gil = pyo3::Python::acquire_gil();
        let py_dict = PyDict::new(gil.python());
        for (key, value) in filtered_scores {
            py_dict.set_item(key, value)?;
        }
        filtered_scores_list.push(py_dict.into());
    }

    Ok(filtered_scores_list)
}

#[pymodule]
fn my_rust_module(_py: Python, m: &PyModule) -> PyResult<()> {
    // Add your Rust functions to the Python module
    m.add_function(wrap_pyfunction!(filter_scores, m)?)?;
    Ok(())
}

fn filter_scores_impl(
    scores_array: &Vec<Vec<f64>>,
    indices_array: &Vec<Vec<usize>>,
    unrolled_names: &Vec<String>,
) -> HashMap<String, f64> {
    let mut filtered_scores: HashMap<String, f64> = HashMap::new();

    // Iterate over query amino scores
    for match_idx in 0..scores_array.len() {
        let match_scores = &scores_array[match_idx];
        let names = indices_array[match_idx]
            .iter()
            .map(|&idx| unrolled_names[idx].clone()) // get target names for each 1000 hits
            .collect::<Vec<_>>();

        let mut sorted_match_idx: Vec<_> = (0..match_scores.len()).collect();
        sorted_match_idx
            .sort_unstable_by(|&a, &b| match_scores[b].partial_cmp(&match_scores[a]).unwrap());

        let mut unique_indices: Vec<_> = Vec::new();
        let mut unique_names: HashMap<String, ()> = HashMap::new();

        for &idx in sorted_match_idx.iter() {
            let name = &names[idx];
            if unique_names.insert(name.clone(), ()).is_some() {
                unique_indices.push(idx);
            }
        }

        let new_indices: Vec<_> = unique_indices
            .iter()
            .map(|&idx| indices_array[match_idx][idx])
            .collect();
        let new_scores: Vec<_> = unique_indices
            .iter()
            .map(|&idx| match_scores[idx])
            .collect();

        for (&distance, name_idx) in new_scores.iter().zip(new_indices.iter()) {
            let name = &unrolled_names[*name_idx];
            *filtered_scores.entry(name.clone()).or_insert(0.0) += distance;
        }
    }

    filtered_scores
}
