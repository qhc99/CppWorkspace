
#include <iostream>

#include "aliases.h"


class Person {
    String name{};
    int age{};
public:
    Person() = default;

    Person(const String &n, const int age) {
        name = n;
        this->age = age;
    }
};


int main() {

    String s{"aaa"};
    Person p{s, 1};
    s.append("a");
    console << "1" << newline;
    Person a{};
    return 0;
}
