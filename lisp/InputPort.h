//
// Created by QC on 2023-01-24.
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

using std::string, std::deque;

class InputPort : public Value {

  deque<string> queue{};
  std::regex pattern{
      R"(\s*(,@|[('`,)]|"(?:[\\].|[^\\"])*"|;.*|[^\s('"`,;)]*))"};
  std::istream *input{};

public:
  /**
   * @brief Construct a new Input Port object
   *
   * @param s lifetime should be longer than InputPort
   */
  explicit InputPort(std::istream &s) : input(&s) {}

  InputPort(const InputPort &other) = default;

  InputPort(InputPort &&other) = default;

  InputPort &operator=(const InputPort &other) = default;

  InputPort &operator=(InputPort &&other) = default;

  /**
   * @return String or EOF Symbol
   */
  shared_ptr<Value> next_token();

  inline string to_string() override { return "#InPort"; }
};

#endif // DEV_QHC_CPP_PROJECTS_INPUTPORT_H
