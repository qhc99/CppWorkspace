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

    inline operator int() const { // NOLINT(google-explicit-constructor)
        return val;
    }
};

class Double : public Value {
public:
    double val{};

    inline operator double() const { // NOLINT(google-explicit-constructor)
        return val;
    }
};

class Complex : public Value {
public:
    std::complex<double> val{};

    inline operator std::complex<double>() const { // NOLINT(google-explicit-constructor)
        return val;
    }
};

class Bool : public Value {
public:
    bool val{};

    inline operator bool() const { // NOLINT(google-explicit-constructor)
        return val;
    }
};

class String : public Value {
public:
    string val{};

    inline String() = default;

    inline String(std::string str) : val(std::move(str)) {} // NOLINT(google-explicit-constructor)

    inline operator string() const { // NOLINT(google-explicit-constructor)
        return val;
    }

    inline String(const String &other) {
        if (this != &other) {
            this->val = other.val;
        }
    }

    inline String(String &&other) noexcept {
        if (this != &other) {
            this->val = std::move(other.val);
        }
    }

    inline String &operator=(const String &other) {
        if (this != &other) {
            this->val = other.val;
        }
        return *this;
    }

    inline String &operator=(String &&other) noexcept {
        if (this != &other) {
            this->val = std::move(other.val);
        }
        return *this;
    }

    inline bool operator==(const String &p) const {
        return val == p.val;
    }
};

inline std::ostream &operator<<(std::ostream &out, const String &str) {
    out << str.val;
    return out;
}

class Symbol : public Value {
public:
    string val{};

    inline Symbol() = default;

    inline explicit Symbol(string str) : val(std::move(str)) {}


    inline Symbol(const Symbol &other) {
        if (&other != this) {
            this->val = other.val;
        }
    };

    inline Symbol(Symbol &&other) noexcept {
        if (&other != this) {
            this->val = std::move(other.val);
        }
    };

    inline Symbol &operator=(const Symbol &other) {
        if (&other != this) {
            this->val = other.val;
        }
        return *this;
    }

    inline Symbol &operator=(Symbol &&other) noexcept {
        if (&other != this) {
            this->val = std::move(other.val);
        }
        return *this;
    }

    inline bool operator==(const Symbol &&other) const {
        return val == other.val;
    }

    inline bool operator!=(const Symbol &&other) const {
        return val != other.val;
    }
};

inline std::ostream &operator<<(std::ostream &out, const Symbol &sym) {
    out << "Symbol(" << sym.val << ")";
    return out;
}


namespace std {
    template<>
    struct hash<String> {
        inline auto operator()(const String &xyz) const -> size_t {
            return hash<string>{}(xyz.val);
        }
    };
}  // namespace std

namespace SYMBOLS {
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

void copy_from(shared_ptr<Value> &ptr, const shared_ptr<Value> &val);

class Pair : public Value {
public:
    shared_ptr<Value> car{};
    shared_ptr<Value> cdr{};


    inline Pair(shared_ptr<Value> car, shared_ptr<Value> cdr) {
        this->car = std::move(car);
        this->cdr = std::move(cdr);
    }

    inline  ~Pair() override = default;

    inline Pair(const Pair &other) {
        if (&other != this) {
            copy_from(this->car, other.car);
            copy_from(this->cdr, other.cdr);
        }
    }

    inline Pair(Pair &&other) noexcept {
        if (&other != this) {
            this->car = other.car;
            this->cdr = other.cdr;
            other.car = nullptr;
            other.cdr = nullptr;
        }
    }

    inline Pair &operator=(const Pair &other) {
        if (&other != this) {
            copy_from(this->car, other.car);
            copy_from(this->cdr, other.cdr);
        }
        return *this;
    }

    inline Pair &operator=(Pair &&other) noexcept {
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
