#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <new>
#include <utility>
#include <vector>

#include <mpi.h>

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

// Exchange ghost rows of buffer u with the up/down neighbours.
//
// TODO: Send your topmost interior row (row 1) to context->up and receive the
// neighbour's border row into the top ghost row (row 0); send your bottommost
// interior row (row context->nloc) to context->down and receive into the bottom
// ghost row (row context->nloc + 1). context->up / context->down are set to
// MPI_PROC_NULL at the domain ends, so those sends/receives become no-ops.
// Use a deadlock-free scheme (MPI_Sendrecv, even/odd ordering, or Isend/Irecv).
JacobiRequests halo_exchange(MPI_Comm comm, std::shared_ptr<GlobalContext> context, double *__restrict__ u)
{
  JacobiRequests reqs;
  MPI_Isend(u + 1 * context->n, context->n, MPI_DOUBLE, context->up, 0, comm, &reqs.send_up);
  MPI_Irecv(u + (context->nloc + 1) * context->n, context->n, MPI_DOUBLE, context->down, 0, comm, &reqs.recv_down);
  MPI_Isend(u + context->nloc * context->n, context->n, MPI_DOUBLE, context->down, 1, comm, &reqs.send_down);
  MPI_Irecv(u, context->n, MPI_DOUBLE, context->up, 1, comm, &reqs.recv_up);
  return reqs;
}

// compute norm of defect on the parallel communicator comm
double defect_norm(MPI_Comm comm, int n, double *__restrict__ u)
{
  double sum = 0.0;

  // TODO: Compute the *squared* local defect norm here. Then reduce it over all ranks
  // with, e.g., MPI_Allreduce and only then take the square root.

  return sqrt(sum);
}

void sync_comm(JacobiRequests requests)
{
  MPI_Wait(&requests.send_up, MPI_STATUS_IGNORE);
  MPI_Wait(&requests.recv_up, MPI_STATUS_IGNORE);
  MPI_Wait(&requests.send_down, MPI_STATUS_IGNORE);
  MPI_Wait(&requests.recv_down, MPI_STATUS_IGNORE);
}

void jacobi_update(double *__restrict__ unew, double *__restrict__ uold, int n, int i0, int i1)
{
  unew[i1 * n + i0] = 0.25 * (uold[i1 * n + i0 - n] +
                              uold[i1 * n + i0 - 1] +
                              uold[i1 * n + i0 + 1] +
                              uold[i1 * n + i0 + n]);
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
    for (int i1 = 2; i1 <= nloc - 1; ++i1)
    {
      for (int i0 = 1; i0 < n - 1; ++i0)
      {
        jacobi_update(unew, uold, n, i0, i1);
      }
    }

    sync_comm(reqs);

    // Update the border rows that touch the ghost rows
    for (int i0 = 1; i0 < n - 1; ++i0)
    {
      jacobi_update(unew, uold, n, i0, 1);
    }
    for (int i0 = 1; i0 < n - 1; ++i0)
    {
      jacobi_update(unew, uold, n, i0, nloc);
    }

    std::swap(uold, unew);

    // make the final iterate available as u0 again
    context->u0 = uold;
    context->u1 = unew;
  }
}

  int main(int argc, char **argv)
  {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::cout << "Rank " << rank << ": Running parallel Jacobi program with " << size << " ranks\n";

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

    jacobi_kernel(MPI_COMM_WORLD, context);

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
    }

    ::operator delete[](context->u1, std::align_val_t(64));
    ::operator delete[](context->u0, std::align_val_t(64));

    MPI_Finalize();
  }
