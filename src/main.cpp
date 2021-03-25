
#include <iostream>
#include <vector>
#include <string>
#include "utils/utils.h"
#include "Euler/tools.h"

using std::cout, std::endl, std::cin;

using namespace org::nathan::utils;

using namespace org::nathan::Euler;


int main()
{
    bool *long_array = new bool[4'147'483'648];
    cout << long_array[4'000'000'000] << endl;

    std::string s;
    cin >> s;
    cout << "end" << endl;
    delete[] long_array;
    return 0;
}
