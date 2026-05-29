cmake_minimum_required(VERSION 3.15)
project(ex05)
set(CMAKE_CXX_STANDARD 20)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(barrier_test barrier_test.cpp)
find_package(OpenMP REQUIRED)
target_link_libraries(barrier_test GTest::gtest GTest::gtest_main OpenMP::OpenMP_CXX)
