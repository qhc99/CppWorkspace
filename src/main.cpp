

#include "aliases.h"
#include "utils.h"

using namespace org::nathan::aliases;
using namespace org::nathan::utils;


auto add(auto a, auto b) {
    return a + b;
}


int main() {
    cout << add(2, 2.2) << endl;


    return 0;
}
