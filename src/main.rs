use rand::Rng;
fn main() {
    //This will simulate a single hidden layer
    let mut rng = rand::thread_rng();
    let mut activation_in = vec![1.0;100]; //100 activations
    let mut activation_out = vec![0.0;100]; //100 activations
    let mut weights = vec![1.0;100 * 100]; //100 * 100 activations

    for i in 0..100{ //fill activations with random stuff
        activation_in[i] = rng.gen_range(-1.0..1.0);
    }
    for i in 0..(100 * 100){ //fill weights with random stuff
        weights[i] = rng.gen_range(-(1.0/100.0)..(1.0/100.0)); // divide by 100 for norm
    }

    use std::time::Instant;
    let now = Instant::now();
    //THE NORMAL ALGORITHM
    //Starting from the part where we just got done doing the
    //dot product in the layer before this one and havent activation functin yet

    for i in 0..100{ //for every activation
        if activation_in[i] < 0.0{
            activation_in[i] = 0.0;
        }
    }

    for i in 0..100{ //for every activation
        for j in 0..100{ //for every output activation of this hidden layer
            activation_out[j] += activation_in[i] * weights[j + i * 100];
        }
    }

    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}, THIS IS NORMAL ALGO", elapsed);



    let now2 = Instant::now();
    //MY ALGORITHM
    for i in 0..100{ //for every activation
        if activation_in[i] > 0.0{
            for j in 0..100{ //for every output activation of this hidden layer
                activation_out[j] += activation_in[i] * weights[j + i * 100];
            }
        } 
    }
    let elapsed2 = now2.elapsed();
    println!("Elapsed: {:.2?}, THIS IS my ALGO", elapsed2);
}
