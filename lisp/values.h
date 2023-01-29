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
    string val{};

    String() = default;

    String(std::string str) : val(std::move(str)) {} // NOLINT(google-explicit-constructor)

    operator string() const { // NOLINT(google-explicit-constructor)
        return val;
    }

    String(const String& other){
        if(this != &other){
            this->val = other.val;
        }
    }

    String(String&& other) noexcept {
        if(this != &other){
            this->val = std::move(other.val);
        }
    }

    String& operator=(const String& other){
        if(this != &other){
            this->val = other.val;
        }
        return *this;
    }

    String& operator=(String&& other) noexcept {
        if(this != &other){
            this->val = std::move(other.val);
        }
        return *this;
    }

    bool operator==(const String& p) const
    {
        return val == p.val;
    }
};

class Symbol : public Value {
public:
    string val{};

    Symbol() = default;

    explicit Symbol(string  str) : val(std::move(str)) {}


    Symbol(const Symbol& other){
        if(&other != this){
            this->val = other.val;
        }
    };

    Symbol(Symbol&& other) noexcept {
        if(&other != this){
            this->val = std::move(other.val);
        }
    };

    Symbol& operator=(const Symbol& other) {
        if(&other != this){
            this->val = other.val;
        }
        return *this;
    }

    Symbol& operator=(Symbol&& other)  noexcept {
        if(&other != this){
            this->val = std::move(other.val);
        }
        return *this;
    }
};

namespace std {
    template <>
    struct hash<String> {
        auto operator()(const String &xyz) const -> size_t {
            return hash<string>{}(xyz.val);
        }
    };
}  // namespace std

namespace SYMBOLS{
    const Symbol QUOTE_SYM{"quote"};
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
    const std::unordered_map<string, Symbol> QUOTES_MAP = {
        {"'",  QUOTE_SYM},
        {"`",  QUASI_QUOTE_SYM},
        {",",  UNQUOTE_SYM},
        {",@", UNQUOTE_SPLICING_SYM},
    };

    const Symbol APPEND_SYM{"append"};
    const Symbol CONS_SYM{"cons"};
    const Symbol LET_SYM{"let"};
}

void copy_from(shared_ptr<Value> &ptr, const shared_ptr<Value>& val);

class Pair : public Value {
public:
    shared_ptr<Value> car{};
    shared_ptr<Value> cdr{};


    Pair(shared_ptr<Value> car, shared_ptr<Value> cdr) {
        this->car = std::move(car);
        this->cdr = std::move(cdr);
    }

    ~Pair() override = default;

    Pair(const Pair &other) {
        if (&other != this) {
             copy_from(this->car, other.car);
             copy_from(this->cdr, other.cdr);
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
             copy_from(this->car, other.car);
             copy_from(this->cdr, other.cdr);
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
