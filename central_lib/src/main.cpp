//
// Created by QC on 2023-01-14.
//
#include <vector>
#include <iostream>
int main()
{
    int t{2};
    std::vector<int> v{1,2,3};
    v[t*2] = 5;
    std::cout << "out";
    return 0;
}