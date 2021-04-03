
#include <iostream>
#include <vector>
#include <string>
#include "euler/tools.h"
#include "utils/utils.h"
#include <string>

using std::cout, std::endl, std::cin, std::string;

using org::nathan::utils::currentTime, org::nathan::utils::intervalToMilli;


int main()
{
    auto t1 = currentTime();
    std::vector<int> vec(3, 100);

    auto it = vec.begin();
    it = vec.insert(it, 200);

    vec.insert(it, 2, 300);

    // "it" no longer valid, get a new one:
    it = vec.begin();

    std::vector<int> vec2(2, 400);
    vec.insert(it + 2, vec2.begin(), vec2.end());
    auto t2 = currentTime();
    cout << intervalToMilli(t2, t1) << endl;


    return 0;
}
