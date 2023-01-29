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
  InputPort(unique_ptr<std::istream> s) : input(std::move(s)) {}

  /**
   * @return String or EOF Symbol
   */
  shared_ptr<Value> nextToken() {
    string line_cache{};
    while (true) {
      if (!queue.empty()) {
        auto ret{make_shared<String>(queue.back())};
        queue.pop_back();
        return std::move(ret);
      }
      if (input->eof()) {
        return make_shared<Symbol>(SYMBOLS::EOF_SYM);
      }
      std::getline(*input, line_cache);
      if (line_cache.empty()) {
        continue;
      }
      auto match_begin =
          std::sregex_iterator{line_cache.begin(), line_cache.end(), pattern};
      auto match_end = std::sregex_iterator{};

      for (auto i = match_begin; i != match_end; ++i) {
        std::smatch match = *i;
        std::string match_str = match.str();
		queue.push_back(std::move(match_str));
	  };
    }
  }
};

#endif // DEV_QHC_CPP_PROJECTS_INPUTPORT_H
