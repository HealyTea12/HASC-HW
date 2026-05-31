#include <cstdint>

#include <benchmark/benchmark.h>
#include <omp.h>

#include "barrier.hh"

template <typename BarrierType>
static void BenchBarrier(benchmark::State &state)
{
    const int32_t n_threads = 4;
    BarrierType barrier(n_threads);

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n_threads * state.range(0));
    for (auto _ : state)
    {
#pragma omp parallel num_threads(n_threads)
        {
            int tid = omp_get_thread_num();
            for (size_t i{0}; i < state.range(0); i++)
            {
                barrier.wait(tid);
            }
        }
    }
}

template <typename T>
class BarrierBenchmarkFixture : public benchmark::Fixture
{
};

BENCHMARK_TEMPLATE_DEFINE_F(BarrierBenchmarkFixture, Barrier, Barrier)(benchmark::State &state)
{
    BenchBarrier<Barrier>(state);
}
BENCHMARK_REGISTER_F(BarrierBenchmarkFixture, Barrier)->Name("Barrier")->Arg(1000)->UseRealTime();

BENCHMARK_TEMPLATE_DEFINE_F(BarrierBenchmarkFixture, CounterBarrier, CounterBarrier)(benchmark::State &state)
{
    BenchBarrier<CounterBarrier>(state);
}
BENCHMARK_REGISTER_F(BarrierBenchmarkFixture, CounterBarrier)->Name("CounterBarrier")->Arg(1000)->UseRealTime();

BENCHMARK_TEMPLATE_DEFINE_F(BarrierBenchmarkFixture, TreeBarrier, TreeBarrier)(benchmark::State &state)
{
    BenchBarrier<TreeBarrier>(state);
}
BENCHMARK_REGISTER_F(BarrierBenchmarkFixture, TreeBarrier)->Name("TreeBarrier")->Arg(1000)->UseRealTime();
