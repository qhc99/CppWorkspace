//
// Created by Nathan on 2023-01-24.
//

#ifndef DEV_QHC_CPP_PROJECTS_INPUTPORT_H
#define DEV_QHC_CPP_PROJECTS_INPUTPORT_H

#include "values.h"
#include <deque>
#include <fstream>
#include <iostream>
#include <memory>
#include <regex>
#include <string>

using std::string, std::deque, std::unique_ptr, std::make_shared;

class InputPort {

    deque<string> queue{};
    const std::regex pattern{
        R"(\s*(,@|[('`,)]|"(?:[\\].|[^\\"])*"|;.*|[^\s('"`,;)]*))"};
    unique_ptr<std::istream> input{};

public:
    explicit InputPort(unique_ptr<std::istream> s) : input(std::move(s)) {}

    /**
     * @return String or EOF Symbol
     */
    shared_ptr<Value> next_token();
};

#endif // DEV_QHC_CPP_PROJECTS_INPUTPORT_H
