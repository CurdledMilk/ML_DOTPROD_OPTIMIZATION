use std::time::{Duration, Instant};
use rand_distr::{Normal, Distribution};
use rand::{thread_rng, Rng};
use rand::distributions::Uniform;
use ndarray::{Array2, Array1};
use packed_simd::f64x4;
use packed_simd::f64x8;
use rayon::prelude::*;
use sprs::{TriMat, TriMatI, CsMat};
use std::vec::Vec;

fn initialize_vector_from_distribution<F, T>(n: usize, mut dist_fn: F) -> Vec<T>
    where F: FnMut() -> T
{
    return (0..n).map(|_| dist_fn()).collect();
}

fn flat_vector_to_matrix<T>(n: usize, flat_vector: Vec<T>) -> Vec<Vec<T>>
    where T: Clone
{
    return flat_vector
        .chunks(n)
        .map(|chunk| chunk.to_vec())
        .collect();
}

fn relu(vec: &[f64]) -> Vec<f64> {
    vec.iter().map(|&x| if x < 0.0 { 0.0 } else { x }).collect()
}

fn dot_product(vec: &[f64], matrix: &[Vec<f64>]) -> Vec<f64> {
    matrix.iter().map(|row| {
        row.iter().zip(vec).map(|(&item, &v)| item * v).sum()
    }).collect()
}

fn optimized_dot_product(vec: &[f64], matrix: &[Vec<f64>]) -> Vec<f64> {
    matrix.iter().map(|row| {
        row.iter().zip(vec).fold(0.0, |acc, (&item, &v)| {
            if item > 0.0 {
                acc + item * v
            } else {
                acc
            }
        })
    }).collect()
}

fn dot_product_loop(activation_in: &[f64], weights_matrix: &Vec<Vec<f64>>, n: usize) -> Vec<f64> {
    let mut activation_out = vec![0.0; n]; //n activations
    for i in 0..n {
        for j in 0..n {
            activation_out[j] += activation_in[i] * weights_matrix[i][j];
        }
    }
    return activation_out;
}

fn checked_dot_product_loop(activation_in: &Vec<f64>, weights: &Vec<Vec<f64>>, n: usize) -> Vec<f64> {
    let mut activation_out = vec![0.0; n]; //n activations
    for i in 0..n {
        if activation_in[i] > 0.0 {
            for j in 0..n {
                activation_out[j] += activation_in[i] * weights[i][j];
            }
        }
    }
    return activation_out;
}

fn generous_checked_dot_product_loop(activation_in: &[f64], flattened_weights: &Vec<f64>, n: usize) -> Vec<f64> {
    let mut activation_out = vec![0.0; n]; // n activations

    for i in 0..n {
        if activation_in[i] > 0.0 {
            for j in 0..n {
                // Calculate the index for the weights matrix stored in a single Vec
                let index = i * n + j;
                activation_out[j] += activation_in[i] * flattened_weights[index];
            }
        }
    }

    activation_out
}

fn double_checked_dot_product_loop(activation_in: &Vec<f64>, weights: &Vec<Vec<f64>>, n: usize) -> Vec<f64> {
    let mut activation_out = vec![0.0; n]; //n activations
    for i in 0..n {
        if activation_in[i] > 0.0 {
            for j in 0..n {
                if weights[i][j] > 0.0 {
                    activation_out[j] += activation_in[i] * weights[i][j];
                }
            }
        }
    }
    return activation_out;
}

fn dot_product_simd(vec: &[f64], matrix: &Vec<Vec<f64>>) -> Vec<f64> {
    matrix
        .par_iter()
        .map(|row| {
            let mut sum = f64x8::splat(0.0);

            // Process in chunks of 8 for SIMD
            let chunks = row.chunks(8);
            let vec_chunks = vec.chunks(8);

            for (r_chunk, v_chunk) in chunks.zip(vec_chunks) {
                let mut r_padded = [0.0; 8];
                let mut v_padded = [0.0; 8];

                // Manually copying elements to padded arrays
                r_padded.iter_mut().zip(r_chunk).for_each(|(p, &r)| *p = r);
                v_padded.iter_mut().zip(v_chunk).for_each(|(p, &v)| *p = v);

                let r_simd = f64x8::from_slice_unaligned(&r_padded);
                let v_simd = f64x8::from_slice_unaligned(&v_padded);

                sum += r_simd * v_simd;
            }

            sum.sum()
        })
        .collect()
}

fn initialize_sparse_vector<F>(size: usize, zero_prob: f64, mut dist_fn: F) -> Vec<f64>
    where
        F: FnMut() -> f64,
{
    let mut rng = thread_rng();
    let uniform = Uniform::new(0.0, 1.0);
    (0..size)
        .map(|_| {
            if uniform.sample(&mut rng) < zero_prob {
                0.0
            } else {
                dist_fn()
            }
        })
        .collect()
}

fn sparse_dot_product(vec: &[f64], matrix: &CsMat<f64>) -> Vec<f64> {
    let mut result = vec![0.0; matrix.rows()];

    for i in 0..matrix.rows() {
        for (j, &val) in matrix.outer_view(i).unwrap().iter() {
            result[j] += vec[i] * val;
        }
    }

    result
}

fn convert_to_array2(matrix: &Vec<Vec<f64>>) -> Array2<f64> {
    let rows = matrix.len();
    let cols = matrix.first().map_or(0, Vec::len);

    let flattened: Vec<f64> = matrix.into_iter().flatten().cloned().collect();

    return Array2::from_shape_vec((rows, cols), flattened).unwrap();
}

fn convert_to_sparse_matrix(dense_matrix: &[Vec<f64>], n: usize) -> CsMat<f64> {
    // Convert to sparse matrix
    let mut triplet = TriMat::new((n, n));
    for (i, row) in dense_matrix.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            if val != 0.0 {
                triplet.add_triplet(i, j, val);
            }
        }
    }
    return triplet.to_csr();
}

fn calculate_sparsity_matrix(matrix: &CsMat<f64>) -> f64 {
    1.0 - (matrix.nnz() as f64) / ((matrix.rows() * matrix.cols()) as f64)
}

fn calculate_sparsity_vector(vector: &[f64]) -> f64 {
    let zero_count = vector.iter().filter(|&&x| x == 0.0).count();
    zero_count as f64 / vector.len() as f64
}

fn time_algorithm<F, R>(mut func: F) -> (R, Duration)
    where
        F: FnMut() -> R,
{
    let start = Instant::now();
    let result = func();
    let duration = start.elapsed();
    (result, duration)
}

fn run_with_config(
    n: usize,
    activations_dist_func: Box<dyn FnMut() -> f64>,
    weights_dist_func: Box<dyn FnMut() -> f64>,
    weights_sparsity: f64
) {
    let start = Instant::now();
    let weights = flat_vector_to_matrix(
        n,
        initialize_sparse_vector(n * n, weights_sparsity, weights_dist_func),
    );
    let activation_in = initialize_vector_from_distribution(n, activations_dist_func);
    let duration = start.elapsed();

    let csr_weights = convert_to_sparse_matrix(&weights, n);
    let flattened_weights: Vec<_> = weights.clone().into_iter().flat_map(|row| row.into_iter()).collect();

    let vectorized_activation_in = Array1::from_vec(activation_in.clone());
    let vectorized_weights_matrix = convert_to_array2(&weights);

    println!("Setup duration: {:?}", duration);
    println!("Matrix rank: {:?}", n);
    println!("Sparsity of weights matrix: {:?}", calculate_sparsity_matrix(&csr_weights));
    println!("Sparsity of activations: {:?}", calculate_sparsity_vector(&relu(&activation_in)));


    let (_dot_product_out, duration) = time_algorithm(|| dot_product(&relu(&activation_in), &weights));
    println!("Basic Map: {:?}", duration);

    let (_dot_product_out, duration) = time_algorithm(|| optimized_dot_product(&relu(&activation_in), &weights));
    println!("Map with check: {:?}", duration);

    let (_dot_product_out, duration) = time_algorithm(|| dot_product_loop(&relu(&activation_in), &weights, n));
    println!("Basic loop: {:?}", duration);

    let (_dot_product_out, duration) = time_algorithm(|| checked_dot_product_loop(&relu(&activation_in), &weights, n));
    println!("Loop with check: {:?}", duration);

    let (_dot_product_out, duration) = time_algorithm(|| generous_checked_dot_product_loop(&relu(&activation_in), &flattened_weights, n));
    println!("Generous flattened weights loop with check (CurdledMilk's Implementation): {:?}", duration);

    let (_dot_product_out, duration) = time_algorithm(|| double_checked_dot_product_loop(&relu(&activation_in), &weights, n));
    println!("Double checked loop: {:?}", duration);

    let (_dot_product_out, duration) = time_algorithm(|| vectorized_weights_matrix.dot(&vectorized_activation_in.mapv(|x| if x < 0.0 { 0.0 } else { x })));
    println!("Vectorized (Array2.dot): {:?}", duration);

    let (_dot_product_out, duration) = time_algorithm(|| dot_product_simd(&relu(&activation_in), &weights));
    println!("Vectorized SIMD: {:?}", duration);

    let (_dot_product_out, duration) = time_algorithm(|| sparse_dot_product(&relu(&activation_in), &csr_weights));
    println!("Sparse: {:?}", duration);
}

fn main() {
    let n = 10000;
    let normal = Normal::new(0.0, 1.0).unwrap();
    let shifted_normal = Normal::new(0.5, 1.0).unwrap();
    let uniform: Uniform<f64> = Uniform::new(-1.0, 1.0);
    let shifted_uniform = Uniform::new(-0.5, 1.0);
    let very_sparse = 1.0 - (1.0 / (n as f64));
    let not_very_sparse = 0.25;

    println!("\nRunning with a uniform distribution around 0 and low sparsity.\n---");
    run_with_config(
        10000,
        Box::new(move || thread_rng().sample(&uniform)),
        Box::new(move || thread_rng().sample(&uniform)),
        not_very_sparse,
    );

    println!("\nRunning with a uniform distribution around 0.5 and low weights sparsity.\n---");
    run_with_config(
        10000,
        Box::new(move || thread_rng().sample(&shifted_uniform)),
        Box::new(move || thread_rng().sample(&uniform)),
        not_very_sparse,
    );

    println!("\nRunning with a uniform distribution and high weights sparsity.\n---");
    run_with_config(
        10000,
        Box::new(move || thread_rng().sample(&uniform)),
        Box::new(move || thread_rng().sample(&uniform)),
        very_sparse,
    );
}