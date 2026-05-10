#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <new>
#include "time_experiment.hh"

// B = A^T,
// A,B are nxn matrices stored row-major in a 1d array,
// assume A and B are NOT the same matrix


// consecutive write, strided read
void transpose1(int n, double *A, double *B)
{
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      B[i * n + j] = A[j * n + i];
}

// strided write, consecutive read
void transpose2(int n, double *A, double *B)
{
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      B[j * n + i] = A[i * n + j];
}


// blocked consecutive write, strided read
template <int M> // MxM blocks
void transpose3(int n, double *A, double *B)
{
  for (int i = 0; i < n; i += M)   // split i loop in blocks of size M
    for (int j = 0; j < n; j += M) // split j loop in blocks of size M
      for (int ii = i; ii < std::min(i + M, n); ii++)
        for (int jj = j; jj < std::min(j + M, n); jj++)
          B[ii * n + jj] = A[jj * n + ii];
}

// blocked strided write, consecutive read
template <int M> // MxM blocks
void transpose4(int n, double *A, double *B)
{
  for (int i = 0; i < n; i += M)   // split i loop in blocks of size M
    for (int j = 0; j < n; j += M) // split j loop in blocks of size M
      for (int ii = i; ii < std::min(i + M, n); ii++)
        for (int jj = j; jj < std::min(j + M, n); jj++)
          B[jj * n + ii] = A[ii * n + jj];
}

// initialize square matrix
void initialize(int n, double *A)
{
  for (int i = 0; i < n * n; i++)
    A[i] = i;
}

double checksum(int n, const double *B)
{
  double sum = 0.0;
  for (int i = 0; i < n * n; i += std::max(1, n * n / 1024))
    sum += B[i];
  return sum;
}

class Experiment1
{
  int n;
  double *A, *B;

public:
  // construct an experiment
  Experiment1(int n_) : n(n_)
  {
    A = new (std::align_val_t{64}) double[n * n];
    B = new (std::align_val_t{64}) double[n * n];
    initialize(n, A);
    initialize(n, B);
    if (((size_t)A) % 64 != 0)
    {
      std::cout << "Exp1: A not aligned to 64 " << std::endl;
    }
    if (((size_t)B) % 64 != 0)
    {
      std::cout << "Exp1: B not aligned to 64 " << std::endl;
    }
  }
  ~Experiment1()
  {
    delete[] A;
    delete[] B;
  }
  // run an experiment; can be called several times
  void operator() () const
  {
    transpose1(n, A, B);
  }
  const double *result() const
  {
    return B;
  }
  // report number of operations for one run
  double operations() const
  {
    return n * n;
  }
};

class Experiment2
{
  int n;
  double *A, *B;

public:
  // construct an experiment
  Experiment2(int n_) : n(n_)
  {
    A = new (std::align_val_t{64}) double[n * n];
    B = new (std::align_val_t{64}) double[n * n];
    initialize(n, A);
    initialize(n, B);
    if (((size_t)A) % 64 != 0)
    {
      std::cout << "Exp2: A not aligned to 64 " << std::endl;
    }
    if (((size_t)B) % 64 != 0)
    {
      std::cout << "Exp2: B not aligned to 64 " << std::endl;
    }
  }
  ~Experiment2()
  {
    delete[] A;
    delete[] B;
  }
  // run an experiment; can be called several times
  void operator() () const
  {
    transpose2(n, A, B);
  }
  const double *result() const
  {
    return B;
  }
  // report number of operations for one run
  double operations() const
  {
    return n * n;
  }
};

template <int M>
class Experiment3
{
  int n;
  double *A, *B;

public:
  // construct an experiment
  Experiment3(int n_) : n(n_)
  {
    A = new (std::align_val_t{64}) double[n * n];
    B = new (std::align_val_t{64}) double[n * n];
    initialize(n, A);
    initialize(n, B);
    if (((size_t)A) % 64 != 0)
    {
      std::cout << "Exp3: A not aligned to 64 " << std::endl;
    }
    if (((size_t)B) % 64 != 0)
    {
      std::cout << "Exp3: B not aligned to 64 " << std::endl;
    }
  }
  ~Experiment3()
  {
    delete[] A;
    delete[] B;
  }
  // run an experiment; can be called several times
  void operator() () const
  {
    transpose3<M>(n, A, B);
  }
  const double *result() const
  {
    return B;
  }
  // report number of operations for one run
  double operations() const
  {
    return n * n;
  }
};

// conduct experiment X with transposeX
template <int M>
class Experiment4
{
  int n;
  double *A, *B;

public:
  // construct an experiment
  Experiment4(int n_) : n(n_)
  {
    A = new (std::align_val_t{64}) double[n * n];
    B = new (std::align_val_t{64}) double[n * n];
    initialize(n, A);
    initialize(n, B);
    if (((size_t)A) % 64 != 0)
    {
      std::cout << "Exp4: A not aligned to 64 " << std::endl;
    }
    if (((size_t)B) % 64 != 0)
    {
      std::cout << "Exp4: B not aligned to 64 " << std::endl;
    }
  }
  ~Experiment4()
  {
    delete[] A;
    delete[] B;
  }
  // run an experiment; can be called several times
  void operator() () const
  {
    transpose4<M>(n, A, B);
  }
  const double *result() const
  {
    return B;
  }
  // report number of operations for one run
  double operations() const
  {
    return n * n;
  }
};

template <typename Experiment>
void run_experiment(const std::string &name, int n, int block_size, double mintime)
{
  Experiment e(n);
  const auto d = time_experiment(e, mintime);
  const double seconds_per_run = d.second / d.first;
  const double runtime_ms = seconds_per_run * 1e3;
  const double bandwidth =
      e.operations() * 2 * sizeof(double) / seconds_per_run / 1e9;
  const double guard = checksum(n, e.result());

  std::cout << name << "," << n << "," << block_size << ","
            << d.first << "," << std::fixed << std::setprecision(6)
            << runtime_ms << "," << std::setprecision(3) << bandwidth
            << "," << guard << std::endl;
}

void run_blocked_consecutive(int n, int block_size, double mintime)
{
  switch (block_size)
  {
  case 4: run_experiment<Experiment3<4>>("blocked-consecutive-write", n, 4, mintime); break;
  case 8: run_experiment<Experiment3<8>>("blocked-consecutive-write", n, 8, mintime); break;
  case 16: run_experiment<Experiment3<16>>("blocked-consecutive-write", n, 16, mintime); break;
  case 24: run_experiment<Experiment3<24>>("blocked-consecutive-write", n, 24, mintime); break;
  case 32: run_experiment<Experiment3<32>>("blocked-consecutive-write", n, 32, mintime); break;
  case 64: run_experiment<Experiment3<64>>("blocked-consecutive-write", n, 64, mintime); break;
  case 128: run_experiment<Experiment3<128>>("blocked-consecutive-write", n, 128, mintime); break;
  }
}

void run_blocked_strided(int n, int block_size, double mintime)
{
  switch (block_size)
  {
  case 4: run_experiment<Experiment4<4>>("blocked-strided-write", n, 4, mintime); break;
  case 8: run_experiment<Experiment4<8>>("blocked-strided-write", n, 8, mintime); break;
  case 16: run_experiment<Experiment4<16>>("blocked-strided-write", n, 16, mintime); break;
  case 24: run_experiment<Experiment4<24>>("blocked-strided-write", n, 24, mintime); break;
  case 32: run_experiment<Experiment4<32>>("blocked-strided-write", n, 32, mintime); break;
  case 64: run_experiment<Experiment4<64>>("blocked-strided-write", n, 64, mintime); break;
  case 128: run_experiment<Experiment4<128>>("blocked-strided-write", n, 128, mintime); break;
  }
}

// main function runs the parameter study and outputs CSV
int main(int argc, char **argv)
{
  const double mintime = argc > 1 ? std::atof(argv[1]) : 0.15;
  const std::vector<int> sizes = {256, 512, 1024, 2048, 4096};
  const std::vector<int> block_sizes = {4, 8, 16, 24, 32, 64, 128};

  std::cout << "strategy,N,block_size,repetitions,runtime_ms,bandwidth_GBps,checksum"
            << std::endl;

  for (int n : sizes)
  {
    run_experiment<Experiment1>("consecutive-write", n, 0, mintime);
    run_experiment<Experiment2>("strided-write", n, 0, mintime);
    for (int block_size : block_sizes)
    {
      run_blocked_consecutive(n, block_size, mintime);
      run_blocked_strided(n, block_size, mintime);
    }
  }

  return 0;
}
