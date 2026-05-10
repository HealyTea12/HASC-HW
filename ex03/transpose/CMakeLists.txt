cmake_minimum_required(VERSION 3.16)
project(transpose_study LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(transpose_study transpose.cc)

target_compile_options(transpose_study PRIVATE
  $<$<CONFIG:Release>:-O3>
  $<$<CONFIG:Release>:-march=native>
  -Wall
  -Wextra
  -pedantic
)
