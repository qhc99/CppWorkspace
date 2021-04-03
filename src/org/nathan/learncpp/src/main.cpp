
#include <iostream>
#include <vector>
#include <string>
#include "euler/tools.h"
#include "utils/utils.h"

using std::cout, std::endl, std::cin, std::string;

using org::nathan::utils::currentTime, org::nathan::utils::intervalToMilli;

using org::nathan::Euler::sieveOfEratosthenes;


int main()
{
    auto t1 = currentTime();
    sieveOfEratosthenes(100000);
    auto t2 = currentTime();

    cout << intervalToMilli(t2, t1) << endl;


    return 0;
}
