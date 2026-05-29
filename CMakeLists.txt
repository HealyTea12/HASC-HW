cmake_minimum_required(VERSION 3.15)
project( assignments )
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_DEBUG_FLAGS "${CMAKE_CXX_DEBUG_FLAGS} -fsanitize=address")
set(CMAKE_CXX_RELEASE_FLAGS "${CMAKE_CXX_RELEASE_FLAGS} -O3 -march=native")

#google benchmark
find_package(benchmark REQUIRED)

add_executable(ex02 ex02/main.cpp ex02/fs.cpp)
target_include_directories(ex02
PRIVATE
    ex02
)

add_executable(bench 
    ex02/benchmark.cpp
    ex02/fs.cpp)
target_link_libraries(bench
    benchmark::benchmark
)

add_subdirectory(ex05)