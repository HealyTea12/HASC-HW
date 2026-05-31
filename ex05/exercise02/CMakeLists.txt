cmake_minimum_required(VERSION 3.15)
project(ex05_exercise02 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(nbody_cpp acceleration_blocked_cppthreads_SoA.cc)
target_link_libraries(nbody_cpp PRIVATE Threads::Threads OpenMP::OpenMP_CXX)
target_include_directories(nbody_cpp PRIVATE ../../ex03/transpose)

target_compile_options(nbody_cpp PRIVATE
  $<$<CONFIG:Release>:-O3>
  $<$<CONFIG:Release>:-march=native>
)
