extern crate packed_simd;
extern crate rayon;

use std::time::Instant;
use rand_distr::{Normal, Distribution};
use rand::{thread_rng, Rng};
use rand::distributions::Uniform;
use ndarray::{Array2, Array1};
use packed_simd::f64x4;
use packed_simd::f64x8;
use rayon::prelude::*;
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
    let mut activation_out = vec![0.0;n]; //n activations
    for i in 0..n {
        for j in 0..n {
            activation_out[j] += activation_in[i] * weights_matrix[i][j];
        }
    }
    return activation_out;
}

fn optimized_dot_product_loop(activation_in: &Vec<f64>, weights: &Vec<Vec<f64>>, n: usize) -> Vec<f64> {
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


fn convert_to_array2(matrix: &Vec<Vec<f64>>) -> Array2<f64> {
    let rows = matrix.len();
    let cols = matrix.first().map_or(0, Vec::len);

    let flattened: Vec<f64> = matrix.into_iter().flatten().cloned().collect();

    return Array2::from_shape_vec((rows, cols), flattened).unwrap();
}

fn main() {
    let start = Instant::now();
    let n = 10000;
    let normal = Normal::new(0.0, 1.0).unwrap();
    let uniform: Uniform<f64> = Uniform::new(-1.0, 1.0);

    let weights_normal = flat_vector_to_matrix(
        n,
        initialize_vector_from_distribution(n * n, || thread_rng().sample(&normal)),
    );
    let weights_uniform = flat_vector_to_matrix(
        n,
        initialize_vector_from_distribution(n * n, || thread_rng().sample(&uniform)),
    );
    let activation_in = initialize_vector_from_distribution(n, || thread_rng().sample(&normal));
    let duration = start.elapsed();
    let weights = weights_normal;
    println!("Setup duration: {:?}", duration);


    let start = Instant::now();
    let relu_out = relu(&activation_in);
    let _dot_product_out = dot_product(&relu_out, &weights);
    let duration = start.elapsed();
    println!("Dot product duration: {:?}", duration);

    let start = Instant::now();
    let relu_out = relu(&activation_in);
    let _dot_product_out = optimized_dot_product(&relu_out, &weights);
    let duration = start.elapsed();
    println!("Optimized dot product duration: {:?}", duration);

    let start = Instant::now();
    let relu_out = relu(&activation_in);
    let _dot_product_out = dot_product_loop(&relu_out, &weights, n);
    let duration = start.elapsed();
    println!("Dot product loop duration: {:?}", duration);

    let start = Instant::now();
    let relu_out = relu(&activation_in);
    let _dot_product_out = optimized_dot_product_loop(&relu_out, &weights, n);
    let duration = start.elapsed();
    println!("Optimized dot product loop duration: {:?}", duration);

    let vectorized_activation_in = Array1::from_vec(activation_in.clone());
    let vectorized_weights_matrix = convert_to_array2(&weights);
    let start = Instant::now();
    let relu_out = vectorized_activation_in.mapv(|x| if x < 0.0 { 0.0 } else { x });
    let _dot_product_out = vectorized_weights_matrix.dot(&relu_out);
    let duration = start.elapsed();
    println!("Vectorized dot product duration: {:?}", duration);

    let start = Instant::now();
    let relu_out = relu(&activation_in);
    let _dot_product_out = dot_product_simd(&relu_out, &weights);
    let duration = start.elapsed();
    println!("Vectorized SIMD dot product duration: {:?}", duration);
}