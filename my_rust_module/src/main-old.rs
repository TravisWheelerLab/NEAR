use rand::distributions::{Distribution, Uniform};
use std::collections::HashMap;

fn filter_scores(
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

fn main() {
    // Generate random data
    let mut rng = rand::thread_rng();
    let mut scores_array: Vec<Vec<f64>> = Vec::new();
    let mut indices_array: Vec<Vec<usize>> = Vec::new();
    let mut unrolled_names: Vec<String> = Vec::new();
    let score_range = Uniform::new_inclusive(0.0, 1.0);
    let index_range = Uniform::new_inclusive(0, 499);
    let name_range = Uniform::new_inclusive(0, 75);

    for _ in 0..500 {
        let mut scores_row: Vec<f64> = Vec::new();
        let mut indices_row: Vec<usize> = Vec::new();

        for _ in 0..1000 {
            scores_row.push(score_range.sample(&mut rng));
            indices_row.push(index_range.sample(&mut rng));
        }

        scores_array.push(scores_row);
        indices_array.push(indices_row);
    }

    for _ in 0..500 {
        unrolled_names.push(format!("Name{}", name_range.sample(&mut rng)));
    }

    // Call the filter_scores function
    let filtered_scores = filter_scores(&scores_array, &indices_array, &unrolled_names);

    // Print the filtered scores for verification
    for (name, distance) in &filtered_scores {
        println!("Name: {}, Distance: {}", name, distance);
    }
}
