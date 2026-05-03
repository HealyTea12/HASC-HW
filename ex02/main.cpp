#include <iostream>
#include <cmath>
#include <fstream>
#include <experimental/simd>

#include "timer.hpp"
#include "integrator.hpp"
#include "fs.hpp"

int main()
{
    namespace stdx = std::experimental;
    const int n = std::pow(2, 30);
    std::cout << "Number of vector lanes used: " << stdx::native_simd<double>::size() << std::endl;

    auto file = std::ofstream("results.txt");
    file << "n, f1, f2, f1_v, f2_v" << std::endl;
    for (int n_ = 8; n_ <= n; n_ *= 2)
    {
        file << n_ << ", ";
        {
            Timer t{
                std::cout,
                [&file](int64_t duration)
                { file << duration << ", "; }};
            std::cout << "f1: " << integrate(f1, 0.0, 1.0, n_) << std::endl;
        }
        {
            Timer t{std::cout, [&file](int64_t duration)
                    { file << duration << ", "; }};
            std::cout << "f2: " << integrate(f2, 0.0, 1.0, n_) << std::endl;
        }
        {
            Timer t{std::cout, [&file](int64_t duration)
                    { file << duration << ", "; }};
            std::cout << "f1_v: " << integrate_v(f1_v, 0.0, 1.0, n_) << std::endl;
        }
        {
            Timer t{std::cout, [&file](int64_t duration)
                    { file << duration << std::endl; }};
            std::cout << "f2_v: " << integrate_v(f2_v, 0.0, 1.0, n_) << std::endl;
        }
    }
}