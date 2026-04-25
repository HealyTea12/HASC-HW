cmake_minimum_required(VERSION 3.10)
project(basic_benchmarks)
set(CMAKE_CXX_STANDARD 20)

find_package(benchmark REQUIRED)
find_package(Threads REQUIRED)

add_executable(basic_benchmarks main.cpp)
add_executable(daxpy daxpy.cpp)

target_link_libraries(basic_benchmarks PRIVATE benchmark::benchmark benchmark::benchmark_main)
target_link_libraries(daxpy PRIVATE Threads::Threads)
