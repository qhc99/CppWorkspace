//
// Created by Nathan on 2021/4/15.
//

#include "demos/templateDemo.h"

template<typename T0, typename... T>
void printf2(T0 t0, T... t) {
    cout << t0 << endl;
    if constexpr (sizeof...(t) > 0) printf2(t...);
    cout << t0 << endl;
}

double add(int a, double b) {
    return a + b;
}

void templateDemo() {
    auto add = [](auto &&PH1, auto &&PH2) {
        return ::add(std::forward<decltype(PH2)>(PH2), std::forward<decltype(PH1)>(PH1));
    };

    cout << add(2.0, 1) << endl;
}