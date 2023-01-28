//
// Created by Nathan on 2023-01-24.
//

#ifndef DEV_QHC_CPP_PROJECTS_INPUTPORT_H
#define DEV_QHC_CPP_PROJECTS_INPUTPORT_H

#include <string>
#include <deque>
#include "values.h"

using std::string,std::deque;

class InputPort {
    deque<string> queue{};
    const string &pattern{R"(\s*(,@|[('`,)]|"(?:[\\].|[^\\"])*"|;.*|[^\s('"`,;)]*))"};
public:
    /**
     * @return String or EOF Symbol
     */
    Value nextToken(){

    }
};


#endif //DEV_QHC_CPP_PROJECTS_INPUTPORT_H
