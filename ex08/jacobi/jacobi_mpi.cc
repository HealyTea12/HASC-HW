#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <new>
#include <vector>

#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif

struct GlobalContext
{
  // input data
  int n;          // nxn global lattice of points including boundary
  int iterations; // number of iterations to do

  // distributed decomposition (1D, contiguous row strips)
  int rank = 0;             // this rank
  int size = 1;             // number of ranks
  int nloc = 0;             // number of interior rows owned by this rank
  int row_offset = 0;       // global index of this rank's first interior row
  int up = MPI_PROC_NULL;   // neighbour rank above (smaller row indices)
  int down = MPI_PROC_NULL; // neighbour rank below (larger row indices)

  // local buffers, each of size (nloc + 2) * n:
  //   row 0        : top ghost row
  //   rows 1..nloc : owned interior rows
  //   row nloc + 1 : bottom ghost row
  double *u0 = nullptr; // the initial guess
  double *u1 = nullptr; // temporary vector

  // output data

  GlobalContext(int n_)
      : n(n_)
  {
  }
  GlobalContext(int n_, int iterations_)
      : n(n_), iterations(iterations_)
  {
  }
  GlobalContext(int n_, int iterations_, int rank_, int size_)
      : n(n_), iterations(iterations_), rank(rank_), size(size_)
  {
  }
};

struct JacobiRequests
{
  MPI_Request send_up;
  MPI_Request recv_up;
  MPI_Request send_down;
  MPI_Request recv_down;
};

// Exchange ghost rows with non-blocking communication.
JacobiRequests halo_exchange(MPI_Comm comm, std::shared_ptr<GlobalContext> context, double *__restrict__ u)
{
  JacobiRequests reqs;
  MPI_Isend(u + 1 * context->n, context->n, MPI_DOUBLE, context->up, 0, comm, &reqs.send_up);
  MPI_Irecv(u + (context->nloc + 1) * context->n, context->n, MPI_DOUBLE, context->down, 0, comm, &reqs.recv_down);
  MPI_Isend(u + context->nloc * context->n, context->n, MPI_DOUBLE, context->down, 1, comm, &reqs.send_down);
  MPI_Irecv(u, context->n, MPI_DOUBLE, context->up, 1, comm, &reqs.recv_up);
  return reqs;
}

void sync_comm(JacobiRequests requests)
{
  MPI_Wait(&requests.send_up, MPI_STATUS_IGNORE);
  MPI_Wait(&requests.recv_up, MPI_STATUS_IGNORE);
  MPI_Wait(&requests.send_down, MPI_STATUS_IGNORE);
  MPI_Wait(&requests.recv_down, MPI_STATUS_IGNORE);
}

void halo_exchange_blocking(MPI_Comm comm, std::shared_ptr<GlobalContext> context, double *__restrict__ u)
{
  const int n = context->n;

  MPI_Sendrecv(u + n, n, MPI_DOUBLE, context->up, 0,
               u + (context->nloc + 1) * n, n, MPI_DOUBLE, context->down, 0,
               comm, MPI_STATUS_IGNORE);
  MPI_Sendrecv(u + context->nloc * n, n, MPI_DOUBLE, context->down, 1,
               u, n, MPI_DOUBLE, context->up, 1,
               comm, MPI_STATUS_IGNORE);
}

// compute norm of defect on the parallel communicator comm
double defect_norm(MPI_Comm comm, std::shared_ptr<GlobalContext> context, double *__restrict__ u)
{
  JacobiRequests reqs = halo_exchange(comm, context, u);
  sync_comm(reqs);

  const int n = context->n;
  double local_sum = 0.0;

  for (int i1 = 1; i1 <= context->nloc; ++i1)
  {
    for (int i0 = 1; i0 < n - 1; ++i0)
    {
      double d = 4.0 * u[i1 * n + i0] -
                 (u[i1 * n + i0 - n] +
                  u[i1 * n + i0 - 1] +
                  u[i1 * n + i0 + 1] +
                  u[i1 * n + i0 + n]);
      local_sum += d * d;
    }
  }

  double global_sum = 0.0;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);

  return sqrt(global_sum);
}

double defect_norm_hybrid(MPI_Comm comm, std::shared_ptr<GlobalContext> context, double *__restrict__ u)
{
  JacobiRequests reqs = halo_exchange(comm, context, u);
  sync_comm(reqs);

  const int n = context->n;
  double local_sum = 0.0;

#pragma omp parallel for reduction(+ : local_sum) schedule(static)
  for (int i1 = 1; i1 <= context->nloc; ++i1)
  {
    for (int i0 = 1; i0 < n - 1; ++i0)
    {
      double d = 4.0 * u[i1 * n + i0] -
                 (u[i1 * n + i0 - n] +
                  u[i1 * n + i0 - 1] +
                  u[i1 * n + i0 + 1] +
                  u[i1 * n + i0 + n]);
      local_sum += d * d;
    }
  }

  double global_sum = 0.0;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);

  return sqrt(global_sum);
}

void jacobi_update(double *__restrict__ unew, double *__restrict__ uold, int n, int i0, int i1)
{
  unew[i1 * n + i0] = 0.25 * (uold[i1 * n + i0 - n] +
                              uold[i1 * n + i0 - 1] +
                              uold[i1 * n + i0 + 1] +
                              uold[i1 * n + i0 + n]);
}

void update_rows(double *__restrict__ unew, double *__restrict__ uold, int n, int first, int last)
{
  if (first > last)
    return;

  for (int i1 = first; i1 <= last; ++i1)
  {
    for (int i0 = 1; i0 < n - 1; ++i0)
      jacobi_update(unew, uold, n, i0, i1);
  }
}

void update_rows_omp(double *__restrict__ unew, double *__restrict__ uold, int n, int first, int last)
{
  if (first > last)
    return;

#pragma omp parallel for schedule(static)
  for (int i1 = first; i1 <= last; ++i1)
  {
    for (int i0 = 1; i0 < n - 1; ++i0)
      jacobi_update(unew, uold, n, i0, i1);
  }
}

double defect_norm_sequential(int n, double *__restrict__ u)
{
  double sum = 0.0;
  for (int i1 = 1; i1 < n - 1; ++i1)
    for (int i0 = 1; i0 < n - 1; ++i0)
    {
      double d = 4.0 * u[i1 * n + i0] -
                 (u[i1 * n + i0 - n] +
                  u[i1 * n + i0 - 1] +
                  u[i1 * n + i0 + 1] +
                  u[i1 * n + i0 + n]);
      sum += d * d;
    }
  return sqrt(sum);
}

void jacobi_kernel_blocking(MPI_Comm comm, std::shared_ptr<GlobalContext> context)
{
  const int n = context->n;
  const int nloc = context->nloc;
  double *uold = context->u0;
  double *unew = context->u1;

  for (int it = 0; it < context->iterations; ++it)
  {
    halo_exchange_blocking(comm, context, uold);
    update_rows(unew, uold, n, 1, nloc);
    std::swap(uold, unew);

    context->u0 = uold;
    context->u1 = unew;
  }
}

// One Jacobi sweep over the local strip, repeated context->iterations times.
// Before each iteration the ghost rows of the source buffer are refreshed via
// halo_exchange, then every owned interior row (1..nloc) is updated.
void jacobi_kernel(MPI_Comm comm, std::shared_ptr<GlobalContext> context)
{
  const int n = context->n;
  const int nloc = context->nloc;
  double *uold = context->u0;
  double *unew = context->u1;

  for (int it = 0; it < context->iterations; ++it)
  {
    // bring the ghost rows of the current iterate up to date
    JacobiRequests reqs = halo_exchange(comm, context, uold);

    // Update interior rows that do not touch the ghost rows
    update_rows(unew, uold, n, 2, nloc - 1);

    sync_comm(reqs);

    // Update the border rows that touch the ghost rows
    update_rows(unew, uold, n, 1, std::min(1, nloc));
    update_rows(unew, uold, n, std::max(2, nloc), nloc);

    std::swap(uold, unew);

    // make the final iterate available as u0 again
    context->u0 = uold;
    context->u1 = unew;
  }
}

void jacobi_kernel_hybrid(MPI_Comm comm, std::shared_ptr<GlobalContext> context)
{
  const int n = context->n;
  const int nloc = context->nloc;
  double *uold = context->u0;
  double *unew = context->u1;

  for (int it = 0; it < context->iterations; ++it)
  {
    JacobiRequests reqs = halo_exchange(comm, context, uold);

    update_rows_omp(unew, uold, n, 2, nloc - 1);

    sync_comm(reqs);

    update_rows_omp(unew, uold, n, 1, std::min(1, nloc));
    update_rows_omp(unew, uold, n, std::max(2, nloc), nloc);

    std::swap(uold, unew);

    context->u0 = uold;
    context->u1 = unew;
  }
}

int main(int argc, char **argv)
{
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::cout << "Rank " << rank << ": Running parallel Jacobi program with " << size << " ranks\n";

    if (provided < MPI_THREAD_FUNNELED)
    {
      if (rank == 0)
        std::cerr << "MPI_THREAD_FUNNELED was requested but not provided\n";
      MPI_Finalize();
      return 1;
    }

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    if (rank == 0)
      std::cout << "MPI thread level: requested FUNNELED, provided " << provided
                << " (MPI calls stay on the main thread), OpenMP threads/rank: "
                << threads << "\n";

    int n = (argc > 1) ? std::atoi(argv[1]) : 512;
    int iterations = (argc > 2) ? std::atoi(argv[2]) : 1000;

    auto context = std::make_shared<GlobalContext>(n, iterations, rank, size);

    const int interior_rows = n - 2;
    const int base_rows = interior_rows / size;
    const int extra_rows = interior_rows % size;

    context->nloc = base_rows + (rank < extra_rows ? 1 : 0);
    context->row_offset = 1 + rank * base_rows + std::min(rank, extra_rows);
    context->up = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    context->down = (rank + 1 < size) ? rank + 1 : MPI_PROC_NULL;

    const int local_size = (context->nloc + 2) * n;
    context->u0 = new (std::align_val_t(64)) double[local_size];
    context->u1 = new (std::align_val_t(64)) double[local_size];

    auto g = [n](int i0, int i1)
    {
      return (i0 > 0 && i0 < n - 1 && i1 > 0 && i1 < n - 1)
                 ? 0.0
                 : static_cast<double>(i0 + i1) / n;
    };

    auto init = [&]()
    {
      for (int i1 = 0; i1 < context->nloc + 2; ++i1)
      {
        const int global_i1 = context->row_offset + i1 - 1;
        double *row0 = context->u0 + i1 * n;
        double *row1 = context->u1 + i1 * n;
        for (int i0 = 0; i0 < n; ++i0)
        {
          const double value = g(i0, global_i1);
          row0[i0] = value;
          row1[i0] = value;
        }
      }
    };

    init();
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    jacobi_kernel_blocking(MPI_COMM_WORLD, context);
    MPI_Barrier(MPI_COMM_WORLD);
    double blocking_time = MPI_Wtime() - start;

    init();
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    jacobi_kernel(MPI_COMM_WORLD, context);
    MPI_Barrier(MPI_COMM_WORLD);
    double overlap_time = MPI_Wtime() - start;

    init();
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    jacobi_kernel_hybrid(MPI_COMM_WORLD, context);
    MPI_Barrier(MPI_COMM_WORLD);
    double hybrid_time = MPI_Wtime() - start;

    double max_blocking_time = 0.0;
    double max_overlap_time = 0.0;
    double max_hybrid_time = 0.0;
    MPI_Reduce(&blocking_time, &max_blocking_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&overlap_time, &max_overlap_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&hybrid_time, &max_hybrid_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double mpi_norm = defect_norm_hybrid(MPI_COMM_WORLD, context, context->u0);

    if (rank == 0)
    {
      const double updates = double(iterations) * (n - 2) * (n - 2);
      std::cout << "blocking:   " << max_blocking_time << " s, "
                << updates / max_blocking_time / 1e9 << " GUpdates/s\n";
      std::cout << "overlapped: " << max_overlap_time << " s, "
                << updates / max_overlap_time / 1e9 << " GUpdates/s\n";
      std::cout << "hybrid:     " << max_hybrid_time << " s, "
                << updates / max_hybrid_time / 1e9 << " GUpdates/s\n";
    }

    std::vector<int> recvcounts;
    std::vector<int> displs;
    std::vector<double> gathered;
    if (rank == 0)
    {
      recvcounts.resize(size);
      displs.resize(size);
      gathered.resize(n * n);
      for (int i1 = 0; i1 < n; ++i1)
        for (int i0 = 0; i0 < n; ++i0)
          gathered[i1 * n + i0] = g(i0, i1);

      for (int r = 0; r < size; ++r)
      {
        const int r_nloc = base_rows + (r < extra_rows ? 1 : 0);
        const int r_row_offset = 1 + r * base_rows + std::min(r, extra_rows);
        recvcounts[r] = r_nloc * n;
        displs[r] = r_row_offset * n;
      }
    }

    MPI_Gatherv(context->u0 + n, context->nloc * n, MPI_DOUBLE,
                rank == 0 ? gathered.data() : nullptr,
                rank == 0 ? recvcounts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);

    if (rank == 0)
    {
      std::vector<double> ref0(n * n);
      std::vector<double> ref1(n * n);
      for (int i1 = 0; i1 < n; ++i1)
        for (int i0 = 0; i0 < n; ++i0)
          ref0[i1 * n + i0] = ref1[i1 * n + i0] = g(i0, i1);

      double *uold = ref0.data();
      double *unew = ref1.data();
      for (int it = 0; it < iterations; ++it)
      {
        for (int i1 = 1; i1 < n - 1; ++i1)
          for (int i0 = 1; i0 < n - 1; ++i0)
            unew[i1 * n + i0] = 0.25 * (uold[i1 * n + i0 - n] +
                                        uold[i1 * n + i0 - 1] +
                                        uold[i1 * n + i0 + 1] +
                                        uold[i1 * n + i0 + n]);
        std::swap(uold, unew);
      }

      double max_diff = 0.0;
      for (int i = 0; i < n * n; ++i)
        max_diff = std::max(max_diff, std::abs(gathered[i] - uold[i]));

      std::cout << "Verify MPI Jacobi: max diff = " << max_diff
                << (max_diff < 1e-12 ? "  [PASS]" : "  [FAIL]") << "\n";

      double ref_norm = defect_norm_sequential(n, uold);
      double norm_diff = std::abs(mpi_norm - ref_norm);
      double norm_tol = 1e-10 * std::max(1.0, ref_norm);
      std::cout << "Defect norm: MPI = " << mpi_norm
                << ", sequential = " << ref_norm
                << ", diff = " << norm_diff
                << (norm_diff < norm_tol ? "  [PASS]" : "  [FAIL]")
                << "\n";
    }

    ::operator delete[](context->u1, std::align_val_t(64));
    ::operator delete[](context->u0, std::align_val_t(64));

    MPI_Finalize();
}
