
#include <iostream>
#include <vector>
#include <string>
#include "euler/tools.h"
#include "utils/utils.h"

using std::cout, std::endl, std::cin, std::string;

using org::nathan::utils::currentTime, org::nathan::utils::timeIntervalToMilli;

using org::nathan::Euler::sieveOfEratosthenes;


int main()
{
    auto t1 = currentTime();
    sieveOfEratosthenes(1000000);
    auto t2 = currentTime();
    cout << timeIntervalToMilli(t2, t1) << endl;


    return 0;
}
