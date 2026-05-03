#include "fs.hpp"
#include <experimental/simd>

namespace stdx = std::experimental;

double f1(double x)
{
    return f_1<double>(x);
};
double f2(double x)
{
    return f_2<double>(x);
};
stdx::native_simd<double> f1_v(stdx::native_simd<double> x)
{
    return f_1<stdx::native_simd<double>>(x);
};
stdx::native_simd<double> f2_v(stdx::native_simd<double> x)
{
    return f_2<stdx::native_simd<double>>(x);
};