#pragma once
#include <chrono>
#include <iostream>
#include <functional>

class Timer
{
    std::ostream &os;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::function<void(int64_t)> callback;

public:
    Timer(std::ostream &os_, std::function<void(int64_t)> cb = nullptr) 
        : os(os_), callback(cb), start(std::chrono::high_resolution_clock::now())
    {
    }
    ~Timer()
    {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        os << "Duration: " << duration << " ms" << std::endl;
        if (callback) {
            callback(duration);
        }
    }
};