#include <benchmark/benchmark.h>

#include <cstdint>
#include <numeric>
#include <vector>

static void BM_EmptyLoopN(benchmark::State &state)
{
    const auto n = state.range(0);
    for (auto _ : state)
    {
        for (int64_t i = 0; i < n; ++i)
        {
        }
    }
}

BENCHMARK(BM_EmptyLoopN)->Arg(1 << 10)->Arg(1 << 20)->Arg(1 << 24);

static void BM_AccumulateN(benchmark::State &state)
{
    const auto n = state.range(0);
    for (auto _ : state)
    {
        int64_t sum = 0;
        for (int64_t i = 0; i < n; ++i)
        {
            sum += i;
        }
        benchmark::DoNotOptimize(sum);
    }
}

BENCHMARK(BM_AccumulateN)->RangeMultiplier(4)->Range(1 << 10, 1 << 20);

static void BM_VectorIotaN(benchmark::State &state)
{
    const auto n = state.range(0);
    for (auto _ : state)
    {
        std::vector<int64_t> values(static_cast<size_t>(n));
        std::iota(values.begin(), values.end(), int64_t{0});
        benchmark::DoNotOptimize(values.data());
    }
}

BENCHMARK(BM_VectorIotaN)->RangeMultiplier(4)->Range(1 << 10, 1 << 20);