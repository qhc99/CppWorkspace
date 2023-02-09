#ifndef QC_CPP_PROJECTS_LISP_FUNC_H
#define QC_CPP_PROJECTS_LISP_FUNC_H
#include "values.h"


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

class Env;

class Procedure : public Func {
public:
  const shared_ptr<Value> exp{};
  const shared_ptr<Value> params{};
  const shared_ptr<Env> env{};

  Procedure(const shared_ptr<Value> &params,
                   const shared_ptr<Value> &exp, const shared_ptr<Env> &env);

  inline string to_string() override { return "#{Procedure}"; }
};
#endif