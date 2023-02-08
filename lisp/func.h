#include "values.h"
#include "Interpreter.h"

class Func : public Value {
protected:
  inline Func() = default;

  std::function<shared_ptr<Value>(shared_ptr<Pair>)> func{};

public:
  inline Func(std::function<shared_ptr<Value>(shared_ptr<Pair>)> f)
      : // NOLINT(google-explicit-constructor)
        func(std::move(f)) {}

  inline shared_ptr<Value> operator()(shared_ptr<Pair> args) const {
    return func(std::move(args));
  }

  inline string to_string() override { return "#{Func}"; }
};


class Procedure : public Func {
public:
  const shared_ptr<Value> exp{};
  const shared_ptr<Value> params{};
  const shared_ptr<Env> env{};

  inline Procedure(const shared_ptr<Value> &params,
                   const shared_ptr<Value> &exp, const shared_ptr<Env> &env)
      : Func(), exp(exp), params(params), env(env) {
    func = [=](const shared_ptr<Pair> &args) {
      return Interpreter::eval(exp, make_shared<Env>(params, args, env));
    };
  }

  inline string to_string() override { return "#{Procedure}"; }
};