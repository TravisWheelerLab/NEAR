use std::collections::HashMap;
use std::ffi::CStr;
use std::ffi::CString;
use std::os::raw::c_char;

#[repr(C)]
pub struct CHashMap {
    pub keys: *mut *const c_char,
    pub values: *const f64,
    pub len: usize,
}

#[no_mangle]
pub extern "C" fn filter_scores(
    scores_array: *const *const f64,
    scores_rows: usize,
    scores_cols: usize,
    indices_array: *const *const c_ulong,
    indices_rows: usize,
    indices_cols: usize,
    unrolled_names: *const *const c_char,
    unrolled_names_len: usize,
) -> CHashMap {
    println!("Convert the raw pointers to Rust slices");

    // Convert the raw pointers to Rust slices
    let scores_array = unsafe { std::slice::from_raw_parts(scores_array, scores_rows) };
    let indices_array = unsafe { std::slice::from_raw_parts(indices_array, indices_rows) };
    let unrolled_names = unsafe { std::slice::from_raw_parts(unrolled_names, unrolled_names_len) };

    println!("Convert the data to Vec<Vec<T>> types");

    // Convert the data to Vec<Vec<T>> types
    let scores_array: Vec<Vec<f64>> = scores_array
        .iter()
        .map(|&ptr| unsafe { std::slice::from_raw_parts(ptr, scores_cols).to_vec() })
        .collect();
    let indices_array: Vec<Vec<usize>> = indices_array
        .iter()
        .map(|&ptr| unsafe { std::slice::from_raw_parts(ptr, indices_cols).to_vec() })
        .collect();
    let unrolled_names: Vec<String> = unrolled_names
        .iter()
        .map(|&ptr| unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() })
        .collect();

    println!("Call the original filter_scores function");

    // Call the original filter_scores function
    let filtered_scores = filter_scores_impl(&scores_array, &indices_array, &unrolled_names);

    println!("Convert the result back to C-compatible format");

    // Convert the result back to C-compatible format
    let mut keys: Vec<*const c_char> = Vec::new();
    let mut values: Vec<f64> = Vec::new();
    for (key, value) in filtered_scores {
        let c_string = CString::new(key).expect("Failed to convert key to C string");
        keys.push(c_string.as_ptr());
        values.push(value);
    }

    CHashMap {
        keys: keys.as_mut_ptr(),
        values: values.as_ptr(),
        len: keys.len(),
    }
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
