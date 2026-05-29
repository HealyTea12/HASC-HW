#include <cstdint>
#include <atomic>
#include <vector>
#include <iostream>

#include <omp.h>
#include <gtest/gtest.h>

#include "barrier.hh"
// Small program that tests that no thread continues until all threads have passed the barrier.

template <typename Barrier>
void test_barrier(Barrier barrier, int32_t n_threads, int32_t n_rounds)
{
    std::vector<std::atomic<int>> arrived(n_rounds);
    for (int32_t k = 0; k < n_rounds; ++k)
        arrived[k].store(0);
    for (int32_t i = 0; i < n_rounds; i++)
    {
#pragma omp parallel num_threads(n_threads)
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            arrived[i]++;
            barrier.wait(tid);
            EXPECT_EQ(arrived[i].load(), nt);
        }
    }
}

template <typename T>
class BarrierTestFixture : public testing::Test
{
};
using BarrierTypes = testing::Types<Barrier, CounterBarrier, TreeBarrier>;
TYPED_TEST_SUITE(BarrierTestFixture, BarrierTypes);
TYPED_TEST(BarrierTestFixture, TestBarrier)
{
    int32_t n_threads = 4;
    int32_t n_rounds = 1000;
    test_barrier(TypeParam(n_threads), n_threads, n_rounds);
};
