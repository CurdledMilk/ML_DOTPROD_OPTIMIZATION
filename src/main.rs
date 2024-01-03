use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use sprs::TriMat;
use std::time::Instant;

fn main() {
    let n = 5000;
    let mut rng = rand::thread_rng();

    // Initialize matrices
    let activation_in = Array2::random_using((1, n), Uniform::new(-1.0, 1.0), &mut rng);
    let weights = Array2::random_using((n, n), Uniform::new(-0.01, 0.01), &mut rng);

    // Measure naive loop algorithm
    let naive_now = Instant::now();
    let mut activation_out = Array2::<f64>::zeros((1, n));
    for i in 0..n {
        for j in 0..n {
            activation_out[[0, j]] += activation_in[[0, i]] * weights[[i, j]];
        }
    }
    println!("Elapsed: {:.2?}, Naive Loop Algorithm", naive_now.elapsed());

    // Measure naive loop algorithm with skipping logic
    let klutzy_now = Instant::now();
    let mut activation_out = Array2::<f64>::zeros((1, n));
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
    let relu_activation_in = activation_in.mapv(|a| if a < 0.0 { 0.0 } else { a });
    let _optimized_activation_out = relu_activation_in.dot(&weights);
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
    let relu_activation_in = activation_in.mapv(|a| if a < 0.0 { 0.0 } else { a });
    let mut sparse_activation_out = vec![0.0; n];
    for i in 0..n {
        if activation_in[[0, i]] > 0.0 {
            for (j, &val) in sparse_weights.outer_view(i).unwrap().iter() {
                sparse_activation_out[j] += relu_activation_in[[0, i]] * val;
            }
        }
    }
    println!("Elapsed: {:.2?}, Sparse Matrix Algorithm", sparse_now.elapsed());

    //RUST compiler automatically vectorizes loops dummy get off your high horse lmao
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut act_in = vec![1.0;n]; //n activations
    let mut act_out = vec![0.0;n]; //n activations
    let mut w = vec![1.0;n * n]; //n * n activations

    for i in 0..n{ //fill activations with random stuff
        act_in[i] = rng.gen_range(-1.0..1.0);
    }
    for i in 0..(n * n){ //fill weights with random stuff
        w[i] = rng.gen_range(-(1.0/(n as f64))..(1.0/(n as f64))); // divide by n for norm
    }
    let now2 = Instant::now();
    //MY ALGORITHM
    for i in 0..n{ //for every activation
        if act_in[i] > 0.0{
            for j in 0..n{ //for every output activation of this hidden layer
                act_out[j] += act_in[i] * w[j + i * n];
            }
        } 
    }
    let elapsed2 = now2.elapsed();
    println!("Elapsed: {:.2?}, THIS IS my old ALGO before some guy ruined it", elapsed2);
}
