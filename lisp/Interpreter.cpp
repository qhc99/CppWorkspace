//
// Created by QC on 2023-01-29.
//

#include "Interpreter.h"
#include "InputPort.h"
#include "values.h"
#include <iostream>
#include <memory.h>
#include <memory>
#include <ostream>
#include <string>

using std::string;

shared_ptr<Value> Interpreter::let(shared_ptr<Pair> args,
                                   Interpreter &interpreter) {
  // TODO unfinished
  return nullptr;
}

shared_ptr<Value> Interpreter::parse(shared_ptr<Value> source,
                                     Interpreter &interpreter) {
  // TODO unfinished
  return std::make_shared<Pair>(nullptr, nullptr);
}

shared_ptr<Value> Interpreter::eval(shared_ptr<Value> x, shared_ptr<Env> env) {
  // TODO unfinished
  return nullptr;
}

void Interpreter::repl() {
  const string prompt{"lisp++ >"};
  auto in_port{make_shared<InputPort>(std::cin)};

  std::cout << ">>>lisp++ interpreter<<<" << std::endl << std::flush;
  while (true) {
    try {
      std::cout << prompt << std::flush;
      auto x{this->parse(in_port)};
      if (x == nullptr) {
        continue;
      } else if (typeid(*x) == typeid(Symbol)) {
        auto syms{std::dynamic_pointer_cast<Symbol>(x)};
        if (SYMBOLS::EOF_SYM == (*syms)) {
          continue;
        }

        auto val{eval(syms, this->global_env)};
        if (val != nullptr) {
          std::cout << val->to_string() << std::flush;
        }
      }
    } catch (BaseException &e) {
      std::cout << e.what() << std::endl << std::flush;
    }
  }
}

void Interpreter::run_file(std::istream file) {
  // TODO unfinished
}

shared_ptr<Value> Interpreter::eval_script() {
  // TODO unfinished
  return nullptr;
}

void Interpreter::load_lib(std::istream file) {
  // TODO unfinished
}

shared_ptr<Value> Interpreter::let(shared_ptr<Pair> args) {
  // TODO unfinished
  return nullptr;
}