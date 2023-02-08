#ifndef QC_CPP_PROJECTS_LISP_ENV
#define QC_CPP_PROJECTS_LISP_ENV
#include "values.h"

class Env : public Value {
public:
    unordered_map<shared_ptr<Value>, shared_ptr<Value>> env{};
    shared_ptr<Env> outer{nullptr};

    inline Env(unordered_map<shared_ptr<Value>, shared_ptr<Value>> e) : // NOLINT(google-explicit-constructor)
        env(std::move(e)) {}

    Env(const shared_ptr<Value> &params, shared_ptr<Pair> args, shared_ptr<Env> outer);

    inline string to_string() override{
        return "#{Env}";
    }

    static Env new_std_env() ;
};

#endif