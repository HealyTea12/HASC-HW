#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace daxpy
{

    void daxpy_parallel(double alpha,
                        const std::vector<double> &x,
                        std::vector<double> &y,
                        std::size_t num_threads = 0)
    {
        const std::size_t n = x.size();

        if (n == 0 || alpha == 0.0)
        {
            return;
        }
        if (x.size() != y.size())
        {
            throw std::invalid_argument("x and y must have the same size");
        }

        std::size_t threads = num_threads;
        if (threads == 0)
        {
            threads = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        }
        threads = std::min<std::size_t>(threads, n);

        if (threads <= 1 || n < 4096)
        {
            for (std::size_t k = 0; k < n; ++k)
            {
                y[k] += alpha * x[k];
            }
            return;
        }

        const std::size_t chunk = (n + threads - 1) / threads;
        std::vector<std::thread> workers;
        workers.reserve(threads);

        for (std::size_t t = 0; t < threads; ++t)
        {
            const std::size_t begin = t * chunk;
            const std::size_t end = std::min(n, begin + chunk);
            if (begin >= end)
            {
                continue;
            }

            workers.emplace_back([&, begin, end, alpha]()
                                 {
			for (std::size_t k = begin; k < end; ++k) {
                y[k] += alpha * x[k];
			} });
        }

        for (auto &th : workers)
        {
            th.join();
        }
    }

} // namespace daxpy

int main()
{
    constexpr std::size_t n = 1 << 20;
    constexpr double alpha = 2.0;

    std::vector<double> x(n);
    std::vector<double> y(n, 1.0);

    for (std::size_t i = 0; i < n; ++i)
    {
        x[i] = static_cast<double>(i);
    }

    daxpy::daxpy_parallel(alpha, x, y);

    std::cout << "y[0]=" << y[0] << " y[n-1]=" << y[n - 1] << '\n';
    return 0;
}
