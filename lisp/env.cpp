
#include "env.h"

Env::Env(const shared_ptr<Value> &params, shared_ptr<Pair> args, shared_ptr<Env> outer) : outer(std::move(outer)) {
    this->outer = outer;
    if (typeid(*params) == typeid(Symbol)) {
        env.insert({params, args});
    } else {
        shared_ptr<Pair> p{std::dynamic_pointer_cast<Pair>(params)};
        while (p != nullptr) {
            env.insert({p->car, args->car});
            p = std::dynamic_pointer_cast<Pair>(p->cdr);
            args = std::dynamic_pointer_cast<Pair>(args->cdr);
        }
        if (args != nullptr) {
            auto msg{"expected: " + params->to_string() + ", " + "given: " + args->to_string()};
            throw TypeException{msg.c_str()};
        }
    }
}