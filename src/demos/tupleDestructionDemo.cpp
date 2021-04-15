//
// Created by Nathan on 2021/3/18.
//

#include "demos/tupleDestructiondemo.h"

namespace org::nathan::tupleDestructionDemo
{

    std::tuple<int, double> returnTuple() // return a tuple that contains an int and a double
    {
        return {5, 6.7};
    }

    [[maybe_unused]] void tupleDestructionDemo()
    {
        auto[a, b]{returnTuple()}; // used structured binding declaration to put results of tuple in variables a and b
        std::cout << a << ' ' << b << '\n';
    }

}