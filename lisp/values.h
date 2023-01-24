//
// Created by Nathan on 2023-01-23.
//

#ifndef DEV_QHC_CPP_PROJECTS_VALUES_H
#define DEV_QHC_CPP_PROJECTS_VALUES_H

#include <typeinfo>
#include <typeindex>
#include <string>
#include <complex>
#include <unordered_map>
#include <memory>
#include <utility>

using std::string, std::shared_ptr;

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
    shared_ptr<string> ptr{};

    String() = default;

    String(const std::string &str) : ptr(std::make_shared<string>(str)) {} // NOLINT(google-explicit-constructor)

    operator shared_ptr<string>() const { // NOLINT(google-explicit-constructor)
        return ptr;
    }

    String(const String& other){
        if(this != &other){
            this->ptr = std::make_shared<string>(*other.ptr);
        }
    }

    String(String&& other) noexcept {
        if(this != &other){
            this->ptr = std::move(other.ptr);
        }
    }

    String& operator=(const String& other){
        if(this != &other){
            this->ptr = std::make_shared<string>(*other.ptr);
        }
        return *this;
    }

    String& operator=(String&& other) noexcept {
        if(this != &other){
            this->ptr = std::move(other.ptr);
        }
        return *this;
    }
};

class Symbol : public Value {
public:
    shared_ptr<string> ptr{};

    Symbol() = default;

    explicit Symbol(const string& str) : ptr(std::make_shared<string>(str)) {}


    explicit Symbol(const shared_ptr<string>& p) : ptr(p) {}

    Symbol(const Symbol& other){
        if(&other != this){
            this->ptr = std::make_shared<string>(*other.ptr);
        }
    };

    Symbol(Symbol&& other) noexcept {
        if(&other != this){
            this->ptr = std::move(other.ptr);

        }
    };

    Symbol& operator=(const Symbol& other) {
        if(&other != this){
            this->ptr = std::make_shared<string>(*other.ptr);
        }
        return *this;
    }

    Symbol& operator=(Symbol&& other)  noexcept {
        if(&other != this){
            this->ptr = std::move(other.ptr);
        }
        return *this;
    }
};

namespace SYMBOLS{
    Symbol QUOTE_SYM{"quote"};
    const Symbol IF_SYM{"if"};
    const Symbol SET_SYM{"set!"};
    const Symbol DEFINE_SYM{"define"};
    const Symbol LAMBDA_SYM{"lambda"};
    const Symbol BEGIN_SYM{"begin"};
    const Symbol DEFINE_MACRO_SYM{"define-macro"};
    const Symbol QUASI_QUOTE_SYM{"quasi-quote"};
    const Symbol UNQUOTE_SYM{"unquote"};
    const Symbol UNQUOTE_SPLICING_SYM{"unquote-splicing"};
    const Symbol EOF_SYM{"#<symbol-eof>"};
    std::unordered_map<string, Symbol> QUOTES_MAP = {
        {"'",  QUOTE_SYM},
        {"`",  QUASI_QUOTE_SYM},
        {",",  UNQUOTE_SYM},
        {",@", UNQUOTE_SPLICING_SYM},
    };

    const Symbol APPEND_SYM{"append"};
    const Symbol CONS_SYM{"cons"};
    const Symbol LET_SYM{"let"};
}

void copy_from(Value **ptr, Value *val);

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
