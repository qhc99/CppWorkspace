//
// Created by Nathan on 2023-01-23.
//

#include "values.h"
void copy_from(Value **ptr, Value *val) {
    const auto &val_type = typeid(*val);
    if (val_type == typeid(Int)) {
        auto *n = new Int{};
        n->val = dynamic_cast<Int *>(val)->val;
        *ptr = n;
    } else if (val_type == typeid(Double)) {
        auto *n = new Double{};
        n->val = dynamic_cast<Double *>(val)->val;
        *ptr = n;
    } else if (val_type == typeid(Complex)) {
        auto *n = new Complex{};
        n->val = dynamic_cast<Complex *>(val)->val;
        *ptr = n;
    }else if (val_type == typeid(Bool)) {
        auto *n = new Bool{};
        n->val = dynamic_cast<Bool *>(val)->val;
        *ptr = n;
    }else if (val_type == typeid(String)) {
        auto *n = new String{};
        n->val = dynamic_cast<String *>(val)->val;
        *ptr = n;
    }else if (val_type == typeid(Symbol)) {
        auto *n = new Symbol{};
        n->val = dynamic_cast<Symbol *>(val)->val;
        *ptr = n;
    }
    else if (val_type == typeid(Pair)) {
        auto *n = new Pair{nullptr,nullptr};
        auto*p{dynamic_cast<Pair *>(val)};
        n->car = p->car;
        n->cdr = p->cdr;
        *ptr = n;
    }
}