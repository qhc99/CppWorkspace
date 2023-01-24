//
// Created by Nathan on 2023-01-23.
//

#ifndef DEV_QHC_CPP_PROJECTS_VALUES_H
#define DEV_QHC_CPP_PROJECTS_VALUES_H

#include <typeinfo>
#include <typeindex>
#include <string>
#include <complex>

class Value {
public:
    virtual ~Value() = default;
};


class Int : public Value {
public:
    int val{};

    operator int() const { // NOLINT(google-explicit-constructor)
        return val;
    }
};

class Double : public Value {
public:
    double val{};

    operator double() const { // NOLINT(google-explicit-constructor)
        return val;
    }
};

class Complex : public Value {
public:
    std::complex<double> val{};

    operator std::complex<double>() const { // NOLINT(google-explicit-constructor)
        return val;
    }
};

class Bool : public Value {
public:
    bool val{};

    operator bool() const { // NOLINT(google-explicit-constructor)
        return val;
    }
};

class String : public Value {
public:
    std::string val{};

    operator std::string() const { // NOLINT(google-explicit-constructor)
        return val;
    }
};

class Symbol : public Value {
public:
    std::string val{};
};

class Pair : public Value {
public:
    Value *car{};
    Value *cdr{};

    static void copy(Value **ptr, Value *val) {
        const auto &val_type = typeid(*val);
        if (val_type == typeid(Int)) {
            auto *n = new Int{};
            n->val = dynamic_cast<Int *>(val)->val;
        } else if (val_type == typeid(Double)) {
            auto *n = new Double{};
            n->val = dynamic_cast<Double *>(val)->val;
        } else if (val_type == typeid(Complex)) {
            auto *n = new Complex{};
            n->val = dynamic_cast<Complex *>(val)->val;
        }

    }

    Pair(Value *car, Value *cdr) {
        this->car = car;
        this->cdr = cdr;
    }

    ~Pair() override {
        delete car;
        delete cdr;
    }

    Pair(const Pair &other) {

    }

    Pair(Pair &&other) noexcept {

    }

    Pair &operator=(const Pair &other) {
        copy(&this->car,other.car);
        copy(&this->cdr,other.cdr);
    }

    Pair &operator=(Pair &&other) {

    }
};

void test() {

}

#endif //DEV_QHC_CPP_PROJECTS_VALUES_H
