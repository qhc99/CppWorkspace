//
// Created by QC on 2023-01-29.
//

#ifndef DEV_QHC_CPP_PROJECTS_INTERPRETER_H
#define DEV_QHC_CPP_PROJECTS_INTERPRETER_H
#include "env.h"
#include "values.h"
#include <memory>
#include <unordered_map>

using std::shared_ptr;
class Func;

class Interpreter final {
private:
  friend class Procedure;
  static shared_ptr<Value> let(shared_ptr<Pair> args, Interpreter &interpreter);

  shared_ptr<Env> global_env{std::make_shared<Env>(Env::new_std_env())};
//   const std::unordered_map<Symbol, Func> macro_table = ;

  static shared_ptr<Value> parse(shared_ptr<Value> source,
                                 Interpreter &interpreter);

  inline shared_ptr<Value> parse(shared_ptr<Value> source) {
    return parse(source, *this);
  }

  static shared_ptr<Value> eval(shared_ptr<Value> x, shared_ptr<Env> env);

public:
  void repl();

  void run_file(std::istream file);

  shared_ptr<Value> eval_script();

  void load_lib(std::istream file);

  shared_ptr<Value> let(shared_ptr<Pair> args);
};

#endif // DEV_QHC_CPP_PROJECTS_INTERPRETER_H
