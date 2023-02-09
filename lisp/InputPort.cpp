//
// Created by QC on 2023-01-24.
//

#include "InputPort.h"
using std::make_shared;
shared_ptr<Value> InputPort::next_token() {
  string line_cache{};
  while (true) {
    if (!queue.empty()) {
      auto ret{make_shared<String>(queue.front())};
      queue.pop_front();
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
      const std::smatch &match = *i;
      std::string match_str = match[1].str();
      if (match_str.rfind(';', 0) != 0 && !match_str.empty()) {
        queue.push_back(std::move(match_str));
      }
    }
  }
}