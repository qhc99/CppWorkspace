#include <iostream>
#include <chrono>

[[maybe_unused]] double steadyIntervalToMilli(
        std::chrono::time_point<std::chrono::steady_clock, std::chrono::duration<double>> a,
        std::chrono::time_point<std::chrono::steady_clock, std::chrono::duration<double>> b) {
    return static_cast<std::chrono::duration<double>>(a - b).count() * 1000;
}



int main() {


    return 0;
}
