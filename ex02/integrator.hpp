#pragma once
#include <experimental/simd>

template <typename Functor>
double integrate(Functor &&f, double a, double b, int n)
{
    double sum = 0.0;
    double h = (b - a) / n;
    double x = a + h / 2.0;
    for (int i = 0; i < n; ++i)
    {
        sum += f(x);
        x += h;
    }
    return sum * h;
}

template <typename Functor>
double integrate_v(Functor &&f, double a, double b, int n)
{
    double sum = 0.0;
    using double_v = std::experimental::simd<double>;
    auto v_size = double_v::size();
    int n_ = (n / v_size) * v_size; // Ensure n is a multiple of the vector size
    double h = (b - a) / n;
    std::experimental::native_simd<double> x{};
    x *= h;
    x += a + h / 2.0;
    for (size_t i = 0; i < n_; i += v_size)
    {
        auto fx = f(x);
        sum += std::experimental::reduce(fx, std::plus{});
        x += h * v_size;
    }
    return sum * h;
}
