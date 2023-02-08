//
// Created by QC on 2023-01-23.
//

#include "values.h"

void copy_from(shared_ptr<Value> &ptr, const shared_ptr<Value> &val) {
    const auto &val_type = typeid(*val);
    if (val_type == typeid(Pair)) {
        auto n = std::make_shared<Pair>(nullptr, nullptr);
        auto val_p{std::dynamic_pointer_cast<Pair>(val)};
        copy_from(n->car, val_p->car);
        copy_from(n->cdr, val_p->cdr);
        ptr = n;
    } else {
        ptr = val;
    }
}



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