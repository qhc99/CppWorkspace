#include <iostream>
#include <chrono>
#include <array>

double steadyIntervalToMilli(
        std::chrono::time_point<std::chrono::steady_clock, std::chrono::duration<double>> a,
        std::chrono::time_point<std::chrono::steady_clock, std::chrono::duration<double>> b){
    return static_cast<std::chrono::duration<double>>(a - b).count()*1000;
}



int main() {

    std::array<int,5> a {1,2,3,4,5};
    for(auto & i : a){
        std::cout << i;
    }
    std::cout << '\n';
    return 0;
}
