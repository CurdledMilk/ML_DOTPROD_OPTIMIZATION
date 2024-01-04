Compares many algorithms for dot product.


In sample runs for a 10000x10000 matrix:
```
    Finished release [optimized] target(s) in 0.01s
     Running `target/release/skull`

Running with a uniform distribution around 0 and low sparsity.
---
Setup duration: 2.888239875s
Matrix rank: 10000
Sparsity of weights matrix: 0.25000287
Sparsity of activations: 0.5024
Basic Map: 89.075125ms
Map with check: 89.728833ms
Basic loop: 65.778959ms
Loop with check: 32.874708ms
Generous flattened weights loop with check (CurdledMilk's Implementation): 23.882042ms
Double checked loop: 174.778542ms
Vectorized (Array2.dot): 52.51025ms
Vectorized SIMD: 9.913083ms
Sparse: 67.333208ms

Running with a uniform distribution around 0.5 and low weights sparsity.
---
Setup duration: 2.7513495s
Matrix rank: 10000
Sparsity of weights matrix: 0.24998083999999998
Sparsity of activations: 0.3363
Basic Map: 88.805792ms
Map with check: 89.560958ms
Basic loop: 66.763834ms
Loop with check: 44.435792ms
Generous flattened weights loop with check (CurdledMilk's Implementation): 33.308334ms
Double checked loop: 233.6535ms
Vectorized (Array2.dot): 51.372667ms
Vectorized SIMD: 9.962958ms
Sparse: 75.083083ms

Running with a uniform distribution and high weights sparsity.
---
Setup duration: 1.349959958s
Matrix rank: 10000
Sparsity of weights matrix: 0.99989769
Sparsity of activations: 0.5027
Basic Map: 88.391ms
Map with check: 89.091209ms
Basic loop: 65.47775ms
Loop with check: 32.888542ms
Generous flattened weights loop with check (CurdledMilk's Implementation): 23.295542ms
Double checked loop: 24.203375ms
Vectorized (Array2.dot): 51.983875ms
Vectorized SIMD: 10.543959ms
Sparse: 70.083µs

Process finished with exit code 0
```
