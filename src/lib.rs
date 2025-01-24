use anyhow::Result;
use ndarray::{ArrayView, Dim};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::{
    collections::HashMap,
    fs::File,
    io::{BufWriter, Write},
    sync::{Arc, Mutex},
};

#[derive(Debug)]
struct QueryInterval {
    pub start: usize,
    pub end: usize,
}

struct Args<'a> {
    scores: ArrayView<'a, f32, Dim<[usize; 2]>>,
    indices: ArrayView<'a, i64, Dim<[usize; 2]>>,
    target_starts: &'a [usize],
    dedup: bool,
    output: Option<Arc<Mutex<Box<dyn Write + Send + Sync>>>>,
}

// --------------------------------------------------
#[pyfunction]
fn process_hits_py(
    _py: Python,
    scores: PyReadonlyArray2<f32>,
    indices: PyReadonlyArray2<i64>,
    query_start_p: PyReadonlyArray1<i64>,
    target_start_p: PyReadonlyArray1<i64>,
    output_path: String,
    num_threads: usize,
) -> PyResult<()> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();

    let output: Arc<Mutex<Box<dyn Write + Send + Sync>>> = Arc::new(
        Mutex::new(Box::new(BufWriter::new(File::create(output_path)?))),
    );
    let scores_array = scores.as_array();
    let indices_array = indices.as_array();
    let query_starts = query_start_p.as_array();

    let target_starts: Vec<usize> = target_start_p
        .as_array()
        .into_iter()
        .flat_map(|v| usize::try_from(*v))
        .collect();

    let mut query_lens: Vec<_> = query_starts
        .iter()
        .zip(query_starts.iter().skip(1))
        .map(|(a, b)| b - a)
        .collect();

    let num_rows = &scores_array.shape()[0];
    let last_len = *num_rows - *query_starts.last().unwrap() as usize;
    query_lens.push(last_len as i64);

    let query_intervals: Vec<_> = query_starts
        .iter()
        .zip(query_lens)
        .map(|(&start, len)| QueryInterval {
            start: start as usize,
            end: (start + len - 1) as usize,
        })
        .collect();

    if let Err(e) = query_intervals.into_par_iter().enumerate().try_for_each(
        |(query_idx, query_interval)| -> Result<()> {
            run(
                query_idx,
                query_interval,
                Args {
                    scores: scores_array,
                    indices: indices_array,
                    target_starts: &target_starts,
                    dedup: false,
                    output: Some(output.clone()),
                    //output: None,
                },
            )
        },
    ) {
        eprintln!("{e:?}");
    }
    Ok(())
}

// --------------------------------------------------
fn run(
    query_idx: usize,
    query_interval: QueryInterval,
    args: Args,
) -> Result<()> {
    let last_target_idx = args.target_starts.len() - 1;
    let last_target_start = args.target_starts[last_target_idx];
    let num_cols = args.scores.shape()[1];
    let query_len = query_interval.end - query_interval.start + 1;
    let mut results: HashMap<usize, f32> =
        HashMap::with_capacity(query_len * num_cols);

    for q_i in query_interval.start..=query_interval.end {
        let qscores: Vec<_> =
            (0..num_cols).map(|col| args.scores[[q_i, col]]).collect();

        let qindices: Vec<_> = (0..num_cols)
            .flat_map(|col| usize::try_from(args.indices[[q_i, col]]))
            .collect();

        let targets: Vec<_> = qindices
            .iter()
            .map(|v| match args.target_starts.binary_search(v) {
                Ok(p) => p,
                Err(tstart) => {
                    if tstart >= last_target_start {
                        // It fell into the last target
                        last_target_idx
                    } else {
                        // Move it back one place
                        tstart - 1
                    }
                }
            })
            .collect();

        // Create [(target_id, score)]
        let mut tscores: Vec<_> = targets.iter().zip(qscores).collect();
        if args.dedup {
            tscores.sort_by(|a, b| {
                a.0.cmp(b.0).then_with(|| b.1.partial_cmp(&a.1).unwrap())
            });
            tscores.dedup_by(|a, b| a.0 == b.0);
        }

        for (target_id, target_score) in tscores {
            if target_score > 0.0 {
                results
                .entry(*target_id)
                .and_modify(|v| *v += target_score)
                .or_insert(target_score);
            }
        }
    }

    if let Some(output) = args.output {
        for (target_id, score) in &results {
            match output.lock() {
                Ok(mut guard) => writeln!(
                    guard,
                    "{} {} {:.7}",
                    query_idx + 1,
                    target_id + 1,
                    score
                )?,
                Err(e) => panic!("ouch: {e}"),
            }
        }
    }

    Ok(())
}

// --------------------------------------------------
#[pymodule]
fn process_hits(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(process_hits_py, m)?)?;

    Ok(())
}