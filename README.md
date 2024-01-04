Compares many algorithms for dot product.


In sample runs for a 10000x10000 matrix:
```
    Finished release [optimized] target(s) in 0.02s
     Running `target/release/skull`

Running with a uniform distribution around 0 and low sparsity.
---
Setup duration: 2.804159875s
Matrix rank: 10000
Sparsity of weights matrix: 0.25000973000000004
Sparsity of activations: 0.5012
Basic Map: 88.6175ms
Map with check: 89.340209ms
Basic loop: 65.801916ms
Loop with check (CurdledMilk's Implementation): 79.143208ms
Double checked loop: 175.595542ms
Vectorized (Array2.dot): 54.310208ms
Vectorized SIMD: 10.407167ms
Sparse: 56.656125ms

Running with a uniform distribution around 0.5 and low weights sparsity.
---
Setup duration: 2.752812042s
Matrix rank: 10000
Sparsity of weights matrix: 0.24995639999999997
Sparsity of activations: 0.3289
Basic Map: 87.372708ms
Map with check: 87.775541ms
Basic loop: 64.59025ms
Loop with check (CurdledMilk's Implementation): 43.528125ms
Double checked loop: 232.548417ms
Vectorized (Array2.dot): 46.492375ms
Vectorized SIMD: 9.915083ms
Sparse: 59.70175ms

Running with a uniform distribution and high weights sparsity.
---
Setup duration: 1.336552625s
Matrix rank: 10000
Sparsity of weights matrix: 0.99990013
Sparsity of activations: 0.4978
Basic Map: 89.014667ms
Map with check: 89.312625ms
Basic loop: 66.633459ms
Loop with check (CurdledMilk's Implementation): 33.896208ms
Double checked loop: 24.424083ms
Vectorized (Array2.dot): 54.584542ms
Vectorized SIMD: 9.965292ms
Sparse: 73.167µs

Process finished with exit code 0
```