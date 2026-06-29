#ifndef HASC_TIME_EXPERIMENT_HH
#define HASC_TIME_EXPERIMENT_HH

#include <chrono>

auto get_time_stamp()
{
  return std::chrono::high_resolution_clock::now();
}

template <typename T>
double get_duration_seconds(T start, T stop)
{
  auto duration = stop - start;
  auto dcast =
      std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
  return dcast / 1e6;
}

#endif
