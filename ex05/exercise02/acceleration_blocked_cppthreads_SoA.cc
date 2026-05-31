// acceleration_blocked_cppthreads_SoA.cc
//
// Exercise 2 b)(i): parallel n-body acceleration with raw std::thread,
// following the pattern of cppthreads/power_method.cc:
//   - a shared GlobalContext struct carries all data and synchronization,
//   - P threads each run the same worker function indexed by a rank,
//   - each thread allocates its own private aI and aJ buffers,
//   - block rows I are distributed cyclically across threads,
//   - write conflicts on shared aglobal are resolved with one mutex per block,
//   - a Barrier separates the parallel work from the sequential part.
//
// Standalone build:
//   g++ -O3 -std=c++20 -pthread acceleration_blocked_cppthreads_SoA.cc -o nbody_cpp
//   ./nbody_cpp <n> <P>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <new>
#include <random>
#include <thread>
#include <vector>

#include "time_experiment.hh"

#ifdef _OPENMP
#include <omp.h>
#endif

const int B = 128;
const double G = 1.0;
const double epsilon2 = 1E-10;

#ifndef __GNUC__
#define __restrict__
#endif

class Barrier
{
  int P;
  int count;
  int round;
  std::mutex m;
  std::condition_variable cv;

public:
  explicit Barrier(int P_) : P(P_), count(0), round(0) {}

  void wait(int)
  {
    std::unique_lock<std::mutex> lk(m);
    int myround = round;
    if (++count == P)
    {
      count = 0;
      ++round;
      cv.notify_all();
    }
    else
    {
      cv.wait(lk, [&] { return round != myround; });
    }
  }
};

void acceleration_blocked_buffered_SoA(int n, double *__restrict__ x,
                                       double *__restrict__ m,
                                       double *__restrict__ aglobal)
{
  double *aI = new (std::align_val_t(64)) double[3 * B];
  double *aJ = new (std::align_val_t(64)) double[3 * B];

  for (int I = 0; I < n; I += B)
  {
    for (int i = 0; i < 3 * B; ++i)
      aI[i] = 0.0;

    for (int i = I; i < I + B; i++)
      for (int j = i + 1; j < I + B; j++)
      {
        double d0 = x[j] - x[i];
        double d1 = x[n + j] - x[n + i];
        double d2 = x[2 * n + j] - x[2 * n + i];
        double r2 = d0 * d0 + d1 * d1 + d2 * d2 + epsilon2;
        double r = sqrt(r2);
        double invfact = G / (r * r2);
        double factori = m[i] * invfact;
        double factorj = m[j] * invfact;
        aI[i - I] += factorj * d0;
        aI[i - I + B] += factorj * d1;
        aI[i - I + 2 * B] += factorj * d2;
        aI[j - I] -= factori * d0;
        aI[j - I + B] -= factori * d1;
        aI[j - I + 2 * B] -= factori * d2;
      }

    for (int J = I + B; J < n; J += B)
    {
      for (int j = 0; j < 3 * B; ++j)
        aJ[j] = 0.0;

      for (int i = I; i < I + B; i++)
        for (int j = J; j < J + B; j++)
        {
          double d0 = x[j] - x[i];
          double d1 = x[n + j] - x[n + i];
          double d2 = x[2 * n + j] - x[2 * n + i];
          double r2 = d0 * d0 + d1 * d1 + d2 * d2 + epsilon2;
          double r = sqrt(r2);
          double invfact = G / (r * r2);
          double factori = m[i] * invfact;
          double factorj = m[j] * invfact;
          aI[i - I] += factorj * d0;
          aI[i - I + B] += factorj * d1;
          aI[i - I + 2 * B] += factorj * d2;
          aJ[j - J] -= factori * d0;
          aJ[j - J + B] -= factori * d1;
          aJ[j - J + 2 * B] -= factori * d2;
        }

      for (int j = 0; j < B; ++j)
        aglobal[J + j] += aJ[j];
      for (int j = 0; j < B; ++j)
        aglobal[J + j + n] += aJ[j + B];
      for (int j = 0; j < B; ++j)
        aglobal[J + j + 2 * n] += aJ[j + 2 * B];
    }

    for (int i = 0; i < B; ++i)
      aglobal[I + i] += aI[i];
    for (int i = 0; i < B; ++i)
      aglobal[I + i + n] += aI[i + B];
    for (int i = 0; i < B; ++i)
      aglobal[I + i + 2 * n] += aI[i + 2 * B];
  }

  delete[] aI;
  delete[] aJ;
}

struct GlobalContext
{
  int nthreads;
  int n;
  double *x;
  double *m;
  double *aglobal;
  std::vector<std::mutex> mutexes;
  Barrier barrier;
  std::vector<double> elapsed;

  GlobalContext(int P, int n_)
      : nthreads(P), n(n_), x(nullptr), m(nullptr), aglobal(nullptr),
        mutexes(n_ / B), barrier(P), elapsed(P, 0.0)
  {
  }
};

void acceleration_worker(std::shared_ptr<GlobalContext> ctx, int rank)
{
  const int n = ctx->n;
  const int P = ctx->nthreads;
  double *__restrict__ x = ctx->x;
  double *__restrict__ m = ctx->m;
  double *__restrict__ aglobal = ctx->aglobal;

  double *aI = new (std::align_val_t(64)) double[3 * B];
  double *aJ = new (std::align_val_t(64)) double[3 * B];

  ctx->barrier.wait(rank);
  auto t_start = get_time_stamp();

  for (int I = rank * B; I < n; I += P * B)
  {
    for (int i = 0; i < 3 * B; ++i)
      aI[i] = 0.0;

    for (int i = I; i < I + B; i++)
      for (int j = i + 1; j < I + B; j++)
      {
        double d0 = x[j] - x[i];
        double d1 = x[n + j] - x[n + i];
        double d2 = x[2 * n + j] - x[2 * n + i];
        double r2 = d0 * d0 + d1 * d1 + d2 * d2 + epsilon2;
        double r = sqrt(r2);
        double invfact = G / (r * r2);
        double factori = m[i] * invfact;
        double factorj = m[j] * invfact;
        aI[i - I] += factorj * d0;
        aI[i - I + B] += factorj * d1;
        aI[i - I + 2 * B] += factorj * d2;
        aI[j - I] -= factori * d0;
        aI[j - I + B] -= factori * d1;
        aI[j - I + 2 * B] -= factori * d2;
      }

    for (int J = I + B; J < n; J += B)
    {
      for (int j = 0; j < 3 * B; ++j)
        aJ[j] = 0.0;

      for (int i = I; i < I + B; i++)
        for (int j = J; j < J + B; j++)
        {
          double d0 = x[j] - x[i];
          double d1 = x[n + j] - x[n + i];
          double d2 = x[2 * n + j] - x[2 * n + i];
          double r2 = d0 * d0 + d1 * d1 + d2 * d2 + epsilon2;
          double r = sqrt(r2);
          double invfact = G / (r * r2);
          double factori = m[i] * invfact;
          double factorj = m[j] * invfact;
          aI[i - I] += factorj * d0;
          aI[i - I + B] += factorj * d1;
          aI[i - I + 2 * B] += factorj * d2;
          aJ[j - J] -= factori * d0;
          aJ[j - J + B] -= factori * d1;
          aJ[j - J + 2 * B] -= factori * d2;
        }

      {
        std::lock_guard<std::mutex> lock(ctx->mutexes[J / B]);
        for (int j = 0; j < B; ++j)
          aglobal[J + j] += aJ[j];
        for (int j = 0; j < B; ++j)
          aglobal[J + j + n] += aJ[j + B];
        for (int j = 0; j < B; ++j)
          aglobal[J + j + 2 * n] += aJ[j + 2 * B];
      }
    }

    {
      std::lock_guard<std::mutex> lock(ctx->mutexes[I / B]);
      for (int i = 0; i < B; ++i)
        aglobal[I + i] += aI[i];
      for (int i = 0; i < B; ++i)
        aglobal[I + i + n] += aI[i + B];
      for (int i = 0; i < B; ++i)
        aglobal[I + i + 2 * n] += aI[i + 2 * B];
    }
  }

  ctx->barrier.wait(rank);

  auto t_stop = get_time_stamp();
  ctx->elapsed[rank] = get_duration_seconds(t_start, t_stop);

  delete[] aI;
  delete[] aJ;
}

double acceleration_blocked_cppthreads_SoA(int P, int n,
                                           double *__restrict__ x,
                                           double *__restrict__ m,
                                           double *__restrict__ aglobal)
{
  auto ctx = std::make_shared<GlobalContext>(P, n);
  ctx->x = x;
  ctx->m = m;
  ctx->aglobal = aglobal;

  std::vector<std::thread> threads;
  threads.reserve(P - 1);
  for (int k = 1; k < P; ++k)
    threads.emplace_back(acceleration_worker, ctx, k);
  acceleration_worker(ctx, 0);
  for (auto &t : threads)
    t.join();

  double tmin = *std::min_element(ctx->elapsed.begin(), ctx->elapsed.end());
  double tmax = *std::max_element(ctx->elapsed.begin(), ctx->elapsed.end());
  std::cout << "per-thread elapsed [s]:";
  for (int k = 0; k < P; ++k)
    std::cout << " " << ctx->elapsed[k];
  std::cout << "\n  load balance: min=" << tmin << " max=" << tmax
            << " imbalance=" << (tmax > 0 ? tmax / std::max(tmin, 1e-12) : 1.0)
            << " (1.0 = perfect)\n";
  return tmax;
}

void acceleration_blocked_omp_SoA(int n, double *__restrict__ x,
                                  double *__restrict__ m,
                                  double *__restrict__ aglobal)
{
  std::vector<std::mutex> mutexes(n / B);

#pragma omp parallel firstprivate(n, x, m, aglobal)
  {
    double *aI = new (std::align_val_t(64)) double[3 * B];
    double *aJ = new (std::align_val_t(64)) double[3 * B];

#pragma omp for schedule(dynamic, 1)
    for (int I = 0; I < n; I += B)
    {
      for (int i = 0; i < 3 * B; ++i)
        aI[i] = 0.0;

      for (int i = I; i < I + B; i++)
        for (int j = i + 1; j < I + B; j++)
        {
          double d0 = x[j] - x[i];
          double d1 = x[n + j] - x[n + i];
          double d2 = x[2 * n + j] - x[2 * n + i];
          double r2 = d0 * d0 + d1 * d1 + d2 * d2 + epsilon2;
          double r = sqrt(r2);
          double invfact = G / (r * r2);
          double factori = m[i] * invfact;
          double factorj = m[j] * invfact;
          aI[i - I] += factorj * d0;
          aI[i - I + B] += factorj * d1;
          aI[i - I + 2 * B] += factorj * d2;
          aI[j - I] -= factori * d0;
          aI[j - I + B] -= factori * d1;
          aI[j - I + 2 * B] -= factori * d2;
        }

      for (int J = I + B; J < n; J += B)
      {
        for (int j = 0; j < 3 * B; ++j)
          aJ[j] = 0.0;

        for (int i = I; i < I + B; i++)
          for (int j = J; j < J + B; j++)
          {
            double d0 = x[j] - x[i];
            double d1 = x[n + j] - x[n + i];
            double d2 = x[2 * n + j] - x[2 * n + i];
            double r2 = d0 * d0 + d1 * d1 + d2 * d2 + epsilon2;
            double r = sqrt(r2);
            double invfact = G / (r * r2);
            double factori = m[i] * invfact;
            double factorj = m[j] * invfact;
            aI[i - I] += factorj * d0;
            aI[i - I + B] += factorj * d1;
            aI[i - I + 2 * B] += factorj * d2;
            aJ[j - J] -= factori * d0;
            aJ[j - J + B] -= factori * d1;
            aJ[j - J + 2 * B] -= factori * d2;
          }

        std::lock_guard<std::mutex> lock(mutexes[J / B]);
        for (int j = 0; j < B; ++j)
          aglobal[J + j] += aJ[j];
        for (int j = 0; j < B; ++j)
          aglobal[J + j + n] += aJ[j + B];
        for (int j = 0; j < B; ++j)
          aglobal[J + j + 2 * n] += aJ[j + 2 * B];
      }

      std::lock_guard<std::mutex> lock(mutexes[I / B]);
      for (int i = 0; i < B; ++i)
        aglobal[I + i] += aI[i];
      for (int i = 0; i < B; ++i)
        aglobal[I + i + n] += aI[i + B];
      for (int i = 0; i < B; ++i)
        aglobal[I + i + 2 * n] += aI[i + 2 * B];
    }

    delete[] aI;
    delete[] aJ;
  }
}

int main(int argc, char **argv)
{
  int n = 4096;
  int P = static_cast<int>(std::thread::hardware_concurrency());
  if (P < 1)
    P = 4;
  if (argc >= 2)
    n = std::atoi(argv[1]);
  if (argc >= 3)
    P = std::atoi(argv[2]);

  if (P < 1)
  {
    std::cerr << "P must be at least 1\n";
    return 1;
  }
  if (n % B != 0)
  {
    std::cerr << n << " is not a multiple of the block size B=" << B << "\n";
    return 1;
  }

  std::cout << "n=" << n << "  P=" << P << "  B=" << B
            << "  blocks nB=" << n / B << "\n";

  double *x = new (std::align_val_t(64)) double[3 * n];
  double *m = new (std::align_val_t(64)) double[n];
  double *a_seq = new (std::align_val_t(64)) double[3 * n];
  double *a_cpp = new (std::align_val_t(64)) double[3 * n];
  double *a_omp = new (std::align_val_t(64)) double[3 * n];

  std::mt19937 gen(42);
  std::uniform_real_distribution<double> pos(-1.0, 1.0);
  std::uniform_real_distribution<double> mass(0.5, 1.5);
  for (int i = 0; i < n; i++)
  {
    x[i] = pos(gen);
    x[n + i] = pos(gen);
    x[2 * n + i] = pos(gen);
    m[i] = mass(gen);
  }

  for (int i = 0; i < 3 * n; i++)
    a_seq[i] = 0.0;
  auto s0 = std::chrono::high_resolution_clock::now();
  acceleration_blocked_buffered_SoA(n, x, m, a_seq);
  auto s1 = std::chrono::high_resolution_clock::now();
  double t_seq = std::chrono::duration<double>(s1 - s0).count();

  for (int i = 0; i < 3 * n; i++)
    a_cpp[i] = 0.0;
  double t_cpp = acceleration_blocked_cppthreads_SoA(P, n, x, m, a_cpp);

#ifdef _OPENMP
  omp_set_num_threads(P);
#endif
  for (int i = 0; i < 3 * n; i++)
    a_omp[i] = 0.0;
  auto o0 = std::chrono::high_resolution_clock::now();
  acceleration_blocked_omp_SoA(n, x, m, a_omp);
  auto o1 = std::chrono::high_resolution_clock::now();
  double t_omp = std::chrono::duration<double>(o1 - o0).count();

  double maxabs_cpp = 0.0;
  double maxrel_cpp = 0.0;
  double maxabs_omp = 0.0;
  double maxrel_omp = 0.0;
  for (int i = 0; i < 3 * n; i++)
  {
    double denom = std::abs(a_seq[i]);
    double diff_cpp = std::abs(a_seq[i] - a_cpp[i]);
    maxabs_cpp = std::max(maxabs_cpp, diff_cpp);
    if (denom > 1e-300)
      maxrel_cpp = std::max(maxrel_cpp, diff_cpp / denom);

    double diff_omp = std::abs(a_seq[i] - a_omp[i]);
    maxabs_omp = std::max(maxabs_omp, diff_omp);
    if (denom > 1e-300)
      maxrel_omp = std::max(maxrel_omp, diff_omp / denom);
  }

  double flop = 13.0 * n * (n - 1.0);
  std::cout << "\nsequential: " << t_seq << " s  ("
            << flop / t_seq / 1e9 << " GFLOP/s)\n";
  std::cout << "cppthreads: " << t_cpp << " s  ("
            << flop / t_cpp / 1e9 << " GFLOP/s)   speedup="
            << t_seq / t_cpp << "x\n";
  std::cout << "openmp:     " << t_omp << " s  ("
            << flop / t_omp / 1e9 << " GFLOP/s)   speedup="
            << t_seq / t_omp << "x\n";
  std::cout << "cppthreads max abs/rel difference vs sequential: " << maxabs_cpp
            << " / " << maxrel_cpp << "\n";
  std::cout << "openmp max abs/rel difference vs sequential: " << maxabs_omp
            << " / " << maxrel_omp << "\n";
  std::cout << (maxrel_cpp < 1e-10 && maxrel_omp < 1e-10
                    ? ">>> PASS: both parallel results match sequential\n"
                    : ">>> CHECK: difference larger than expected\n");

  delete[] x;
  delete[] m;
  delete[] a_seq;
  delete[] a_cpp;
  delete[] a_omp;
  return 0;
}
