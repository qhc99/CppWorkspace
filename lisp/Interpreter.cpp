//
// Created by QC on 2023-01-29.
//

#include "Interpreter.h"
#include "InputPort.h"
#include "exceptions.h"
#include "utils.h"
#include "values.h"
#include <iostream>
#include <memory.h>
#include <memory>
#include <ostream>
#include <string>

using std::string;

shared_ptr<Value> Interpreter::let(shared_ptr<Pair> args,
                                   Interpreter &interpreter) {
  
  auto x{std::make_shared<Pair>(std::make_shared<Symbol>(SYMBOLS::LET_SYM))};
  x->cdr = args;
  int args_len{args->lenth()};
  require(x, args_len + 1 > 1);
  auto bindings{std::dynamic_pointer_cast<Pair>(args->car)};
  if (bindings == nullptr) {
    throw SyntaxException{"illegal binding list"};
  }
  auto body{args->cdr};
  require(x, Pair::all_match(bindings, [](shared_ptr<Value> l) -> bool {
            auto pl{std::dynamic_pointer_cast<Pair>(l)};
            if (pl == nullptr)
              return false;
            return pl->lenth() == 2 && typeid(pl->car) == typeid(Symbol);
          }));
  auto vars{Pair::map(bindings, [](shared_ptr<Value> v) -> shared_ptr<Value> {
    auto vl{std::dynamic_pointer_cast<Pair>(v)};
    return vl->car;
  })};
  auto vals{Pair::map(bindings, [](shared_ptr<Value> v) -> shared_ptr<Value> {
    auto vl{std::dynamic_pointer_cast<Pair>(v)};
    return vl->cdr;
  })};
  // TODO unfinished
  return nullptr;
  /*
    List<Object> x = treeList(_let);
    x.add(args);
    require(x, x.size() > 1);
    List<List<Object>> bindings;
    try {
        bindings = (List<List<Object>>) args.get(0);
    } catch (ClassCastException e) {
        throw new ClassCastException("illegal binding list");
    }
    List<Object> body = args.subList(1, args.size());
    require(x, bindings.stream().allMatch(b -> b != null &&
            b.size() == 2 &&
            b.get(0) instanceof Symbol), "illegal binding list");
    List<Object> vars = bindings.stream().map(l -> l.get(0)).
        collect(Collectors.toList());
    List<Object> vals = bindings.stream().map(l -> l.get(1)).toList();
    var t = treeList(_lambda, vars);
    t.addAll(body.stream().map(interpreter::expand).toList());
    var r = treeList(t);
    r.addAll(vals.stream().map(interpreter::expand).toList());
    return r;
  */
}

shared_ptr<Value> Interpreter::parse(shared_ptr<Value> source,
                                     Interpreter &interpreter) {
  // TODO unfinished
  /*
  if (in instanceof String strIn) {
            var t = read(new InputPort(strIn));
            return interpreter.expand(t, true);
        } else if (in instanceof InputPort inPort) {
            var t = read(inPort);
            return interpreter.expand(t, true);
        } else {
            throw new RuntimeException();
        }
  */
  return std::make_shared<Pair>(nullptr, nullptr);
}

shared_ptr<Value> Interpreter::eval(shared_ptr<Value> x, shared_ptr<Env> env) {
  // TODO unfinished
  /*
  while (true) {
            if (x instanceof Symbol) {
                return env.findEnv(x).get(x);
            } else if (!(x instanceof List<?>)) {
                return x;
            }
            List<Object> l = (List<Object>) x;
            var op = l.get(0);
            if (op.equals(_quote)) {
                return l.get(1); // TODO return cons list
            } else if (op.equals(_if)) {
                var test = l.get(1);
                var conseq = l.get(2);
                var alt = l.get(3);
                boolean testBool = isTrue(eval(test, env));
                if (testBool) {
                    x = conseq;
                } else {
                    x = alt;
                }
            } else if (op.equals(_set)) {
                var v = l.get(1);
                var exp = l.get(2);
                env.findEnv(v).put(v, eval(exp, env));
                return null;
            } else if (op.equals(_define)) {
                var v = l.get(1);
                var exp = l.get(2);
                env.put(v, eval(exp, env));
                return null;
            } else if (op.equals(_lambda)) {
                var vars = l.get(1);
                var exp = l.get(2);
                return Procedure.newProcedure(vars, exp, env);
            } else if (op.equals(_begin)) {
                for (var exp : l.subList(1, l.size() - 1)) {
                    eval(exp, env);
                }
                x = l.get(l.size() - 1);
            } else {
                Environment finalEnv = env;
                List<Object> exps = l.stream().map(exp ->
                    eval(exp, finalEnv)).collect(Collectors.toList());
                var proc = exps.get(0);
                exps = exps.subList(1, exps.size());
                if (proc instanceof Procedure p) {
                    x = p.expression();
                    env = new Environment(p.parameters(), exps,
  p.environment()); } else { return ((Lambda) proc).apply(exps);
                }
            }
        }
  */
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
    } catch (const BaseException &e) {
      std::cout << e.what() << std::endl << std::flush;
    }
  }
}

void Interpreter::run_file(std::istream file) {
  auto in_port{make_shared<InputPort>(file)};
  while (true) {
    try {
      auto syms{parse(in_port)};
      if (typeid(*syms) == typeid(Symbol) &&
          *std::dynamic_pointer_cast<Symbol>(syms) == SYMBOLS::EOF_SYM) {
        std::cout << std::flush;
        return;
      }
      auto val{eval(syms, this->global_env)};
      if (val != nullptr) {
        std::cout << val->to_string();
      }
    } catch (const BaseException &e) {
      std::cout << e.what() << std::endl;
    }
  }
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