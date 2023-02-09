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
  // TODO unfinished
  const string prompt{"lisp++>"};
  auto in_port{make_shared<InputPort>(std::cin)};

  std::cout << ">>lisp++ interpreter<<" << std::endl;
  while (true) {
    try {
      std::cout << prompt << std::flush;
      auto x{this->parse(in_port)};
      if (x == nullptr) {
        continue;
      } else if (typeid(*x) == typeid(Symbol)) {
        auto s{std::dynamic_pointer_cast<Symbol>(x)};
        if (SYMBOLS::EOF_SYM == (*s)) {
          continue;
        }

        auto val{eval(s, this->global_env)};
      }
    } catch (BaseException &e) {
    }
  }
  /*
  String prompt = "Jis.py>";
  InputPort inPort = new InputPort(System.in);
  System.out.println("Jispy version 2.0");
  while(true) {
    try{
      System.out.print(prompt);
      var x = parse(inPort);
      if(x == null){ continue; }
      else if(x.equals(eof)){ continue; }
      evalAndPrint(x);
      flushConsole();
    }
    catch(Exception e){
      consoleWriteLine(e.toString());
      flushConsole();
    }
  }
  */
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