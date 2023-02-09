#include "Interpreter.h"
#include "func.h"


Procedure::Procedure(const shared_ptr<Value> &params,
                   const shared_ptr<Value> &exp, const shared_ptr<Env> &env)
      : Func(), exp(exp), params(params), env(env) {
    func = [=](const shared_ptr<Pair> &args) {
      return Interpreter::eval(exp, make_shared<Env>(params, args, env));
    };
  }