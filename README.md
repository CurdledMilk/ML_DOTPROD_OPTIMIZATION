Compares 4 algorithms for dot product:

1. Klutzy's naive for loop algorithm
2. Klutzy's naive for loop with conditional algorithm
3. marr75's vectorized algorithm
4. marr75's sparse matrix vectorized algorithm
5. Klutzy's naive for loop with conditional algorithm (again)

In a sample run for a 10000x10000 matrix:
```
Finished release [optimized] target(s) in 0.01s
Running `target/release/skull`
Setup duration: 3.175579583s
Dot product duration: 108.85375ms
Optimized dot product duration: 89.647875ms
Dot product loop duration: 66.7235ms
Optimized dot product loop duration: 32.958334ms
Vectorized dot product duration: 64.812541ms
Vectorized SIMD dot product duration: 9.849417ms
```