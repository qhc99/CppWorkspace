//
// Created by Nathan on 2023-01-29.
//

#ifndef DEV_QHC_CPP_PROJECTS_INTERPRETER_H
#define DEV_QHC_CPP_PROJECTS_INTERPRETER_H
#include <memory>
#include "values.h"

using std::shared_ptr;

class Interpreter final {
public:
  void repl();

  void run_file(std::istream file);

  shared_ptr<Value> eval_script();

  void load_lib(std::istream file);
};

#endif // DEV_QHC_CPP_PROJECTS_INTERPRETER_H
