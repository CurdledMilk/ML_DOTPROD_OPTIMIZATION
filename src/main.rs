use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use sprs::{CsMat, TriMat};
use std::time::Instant;

fn main() {
    let n = 5000;
    let mut rng = rand::thread_rng();

    // Initialize matrices
    let activation_in = Array2::random_using((1, n), Uniform::new(-1.0, 1.0), &mut rng);
    let weights = Array2::random_using((n, n), Uniform::new(-0.01, 0.01), &mut rng);

    // Measure naive loop algorithm
    let mut activation_out = Array2::<f64>::zeros((1, n));
    let naive_now = Instant::now();
    for i in 0..n {
        for j in 0..n {
            activation_out[[0, j]] += activation_in[[0, i]] * weights[[i, j]];
        }
    }
    println!("Elapsed: {:.2?}, Naive Loop Algorithm", naive_now.elapsed());

    // Measure naive loop algorithm with skipping logic
    let mut activation_out = Array2::<f64>::zeros((1, n));
    let klutzy_now = Instant::now();
    for i in 0..n {
        if activation_in[[0, i]] > 0.0 {
            for j in 0..n {
                activation_out[[0, j]] += activation_in[[0, i]] * weights[[i, j]];
            }
        }
    }
    println!("Elapsed: {:.2?}, Naive Loop w/ Skip Algorithm", klutzy_now.elapsed());

    // Measure vectorized algorithm
    let marr75_now = Instant::now();
    let optimized_activation_out = activation_in.dot(&weights);
    println!("Elapsed: {:.2?}, Vectorized Algorithm", marr75_now.elapsed());

    // Convert the dense matrix to a sparse matrix using sprs
    let mut triplet = TriMat::new((n, n));
    for i in 0..n {
        for j in 0..n {
            let val = weights[[i, j]];
            if val != 0.0 {
                triplet.add_triplet(i, j, val);
            }
        }
    }
    let sparse_weights = triplet.to_csr::<usize>();

    // Measure sparse matrix algorithm
    let sparse_now = Instant::now();
    let mut sparse_activation_out = vec![0.0; n];
    for i in 0..n {
        if activation_in[[0, i]] > 0.0 {
            for (j, &val) in sparse_weights.outer_view(i).unwrap().iter() {
                sparse_activation_out[j] += activation_in[[0, i]] * val;
            }
        }
    }
    println!("Elapsed: {:.2?}, Sparse Matrix Algorithm", sparse_now.elapsed());
}
