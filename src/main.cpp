

#include "aliases.h"

using namespace org::nathan::aliases;


#include <tuple>


std::tuple<int, double> returnTuple() // return a tuple that contains an int and a double
{
    return {5, 6.7};
}

int main() {
    auto[a, b]{returnTuple()}; // used structured binding declaration to put results of tuple in variables a and b
    std::cout << a << ' ' << b << '\n';

    return 0;
}
