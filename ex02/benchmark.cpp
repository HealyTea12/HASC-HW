#include "fs.hpp"
#include "integrator.hpp"

#include <experimental/simd>
#include <benchmark/benchmark.h>
#include <utility>

namespace stdx = std::experimental;

namespace
{
    auto integrate_scalar = [](auto &&func, double a, double b, int n)
    {
        return integrate(std::forward<decltype(func)>(func), a, b, n);
    };

    auto integrate_vectorized = [](auto &&func, double a, double b, int n)
    {
        return integrate_v(std::forward<decltype(func)>(func), a, b, n);
    };

    template <typename Func, typename IntegrateFn>
    void run_benchmark(benchmark::State &state, Func &&func, IntegrateFn &&integrate_fn)
    {
        for (auto _ : state)
        {
            benchmark::DoNotOptimize(integrate_fn(func, 0.0, 1.0, state.range(0)));
        }
    }

    template <typename Func, typename IntegrateFn>
    void benchmark_function(benchmark::State &state, Func &&func, IntegrateFn &&integrate_fn)
    {
        run_benchmark(state, std::forward<Func>(func), std::forward<IntegrateFn>(integrate_fn));
    }
} // namespace

static void BM_f1(benchmark::State &state)
{
    benchmark_function(state, f1, integrate_scalar);
}

static void BM_f2(benchmark::State &state)
{
    benchmark_function(state, f2, integrate_scalar);
}

static void BM_f1_v(benchmark::State &state)
{
    benchmark_function(state, f1_v, integrate_vectorized);
}

static void BM_f2_v(benchmark::State &state)
{
    benchmark_function(state, f2_v, integrate_vectorized);
}

BENCHMARK(BM_f1)->RangeMultiplier(2)->Range(8, 1 << 30);
BENCHMARK(BM_f2)->RangeMultiplier(2)->Range(8, 1 << 30);
BENCHMARK(BM_f1_v)->RangeMultiplier(2)->Range(8, 1 << 30);
BENCHMARK(BM_f2_v)->RangeMultiplier(2)->Range(8, 1 << 30);

BENCHMARK_MAIN();
