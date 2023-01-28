//
// Created by Nathan on 2023-01-23.
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