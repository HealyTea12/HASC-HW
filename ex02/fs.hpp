#pragma once
#include <experimental/simd>

template <typename T>
T f_1(T x)
{
    return x * x * x - 2 * x * x + 3 * x - 1;
};

template <typename T>
T f_2(T x)
{
    T x_ = x;
    for (int i{0}; i < 16; i++)
    {
        x += x_;
        x_ *= x_;
    }
    return x;
};

double f1(double x);
double f2(double x);
std::experimental::native_simd<double> f1_v(std::experimental::native_simd<double> x);
std::experimental::native_simd<double> f2_v(std::experimental::native_simd<double> x);