//
// Created by QC on 2023-01-29.
//

#ifndef DEV_QHC_CPP_PROJECTS_INTERPRETER_H
#define DEV_QHC_CPP_PROJECTS_INTERPRETER_H
#include "values.h"
#include <memory>

using std::shared_ptr;

class Interpreter final {
private:
  static shared_ptr<Value> parse(shared_ptr<Value> source,
                                 Interpreter &interpreter);

  inline shared_ptr<Value> parse(shared_ptr<Value> source) {
    return parse(source, *this);
  }

public:
  void repl();

  void run_file(std::istream file);

  shared_ptr<Value> eval_script();

  void load_lib(std::istream file);
};

#endif // DEV_QHC_CPP_PROJECTS_INTERPRETER_H
