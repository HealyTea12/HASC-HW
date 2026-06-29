# Jacobi comparison

Baseline: `jacobi_omp.cc` implements a plain row-parallel OpenMP Jacobi
kernel. This is used as the pure-thread baseline because the previous sheet's
version was not available.

Build:

```sh
g++ -std=c++17 -O3 -fopenmp jacobi_omp.cc -o jacobi_omp
mpicxx -std=c++17 -O3 -fopenmp jacobi_mpi.cc -o jacobi_mpi
```

Run, fixed total of 4 workers:

```sh
OMP_NUM_THREADS=4 ./jacobi_omp 1024 200
OMP_NUM_THREADS=1 mpirun -np 4 ./jacobi_mpi 1024 200
OMP_NUM_THREADS=2 mpirun -np 2 ./jacobi_mpi 1024 200
```

Results on this machine:

| version | configuration | throughput |
|---|---:|---:|
| sequential | 1 process | 0.735 GUpdates/s |
| OpenMP | 4 threads | 1.152 GUpdates/s |
| MPI blocking | 4 ranks x 1 thread | 0.933 GUpdates/s |
| MPI overlapped | 4 ranks x 1 thread | 1.034 GUpdates/s |
| hybrid | 2 ranks x 2 threads | 0.468 GUpdates/s |

All runs passed the correctness check against the sequential reference. The MPI
defect norm also matched the sequential defect norm.

The pure OpenMP version is fastest for `n=1024` on this machine, but pure MPI
gets close. MPI has extra halo exchange work every iteration, but the overhead
gets less important when the strips are larger. In a size sweep, MPI overlap was
much worse for `n=512`, close for `n=1024` and `n=2048`, and slightly faster for
`n=4096` in one run. This is the surface-to-volume effect: each rank exchanges
roughly two rows, while the amount of local computation grows with the strip
area.
