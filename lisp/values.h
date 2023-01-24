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

void copy_from(Value **ptr, Value *val) ;

class Pair : public Value {
public:
    Value *car{};
    Value *cdr{};



    Pair(Value *car, Value *cdr) {
        this->car = car;
        this->cdr = cdr;
    }

    ~Pair() override {
        delete car;
        delete cdr;
    }

    Pair(const Pair &other) {
        if (&other != this) {
            copy_from(&this->car, other.car);
            copy_from(&this->cdr, other.cdr);
        }
    }

    Pair(Pair &&other) noexcept {
        if (&other != this) {
            this->car = other.car;
            this->cdr = other.cdr;
            other.car = nullptr;
            other.cdr = nullptr;
        }
    }

    Pair &operator=(const Pair &other) {
        if (&other != this) {
            copy_from(&this->car, other.car);
            copy_from(&this->cdr, other.cdr);
        }
        return *this;
    }

    Pair &operator=(Pair &&other) noexcept {
        if (&other != this) {
            this->car = other.car;
            this->cdr = other.cdr;
            other.car = nullptr;
            other.cdr = nullptr;
        }
        return *this;
    }
};

#endif //DEV_QHC_CPP_PROJECTS_VALUES_H
