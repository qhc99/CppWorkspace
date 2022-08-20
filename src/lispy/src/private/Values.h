//
// Created by Nathan on 2022-08-16.
//

#ifndef ORG_QHC_LISP_CPP_VALUES_H
#define ORG_QHC_LISP_CPP_VALUES_H

#include <string>
#include <typeindex>
#include <complex>

struct AnyVal;

using Ptr_Val = std::shared_ptr<const AnyVal>;
using std::shared_ptr, std::string, std::make_shared;

struct AnyVal {
  std::type_index typeIndex{typeid(AnyVal)};

  explicit AnyVal(std::type_index t) : typeIndex(t) {};

  virtual ~AnyVal() = default;
};

struct Bool : public AnyVal {
  bool data{};

  Bool(bool data) : data(data), AnyVal(typeid(Bool)) { // NOLINT(google-explicit-constructor)
  }

  operator bool() const { return data; } // NOLINT(google-explicit-constructor)

};

struct Int : public AnyVal {
  int data{};

  Int(int data) : data(data), AnyVal(typeid(Int)) { // NOLINT(google-explicit-constructor)
  }

  operator int() const { return data; } // NOLINT(google-explicit-constructor)

};

struct Double : public AnyVal {
  double data{};

  Double(double data) : data(data), AnyVal(typeid(Double)) { // NOLINT(google-explicit-constructor)
  }

  operator double() const { return data; } // NOLINT(google-explicit-constructor)

};

struct Complex : public AnyVal {
  std::complex<double> data{};

  Complex(std::complex<double> data) : data(data), AnyVal(typeid(Complex)) { // NOLINT(google-explicit-constructor)
  }

  operator std::complex<double>() const { return data; } // NOLINT(google-explicit-constructor)
};

struct String : public AnyVal {
  string data{};

  String(string data) : data(std::move(data)), AnyVal(typeid(String)) { // NOLINT(google-explicit-constructor)
  }

  operator string() const { return data; } // NOLINT(google-explicit-constructor)

};


struct Tuple : public AnyVal {
  Ptr_Val car{};
  Ptr_Val cdr{};

  Tuple(const Ptr_Val &car, const Ptr_Val &cdr) : AnyVal(typeid(Tuple)) {
    if (car->typeIndex != typeid(Tuple)) {
      // unique ptr is not convenient for polymorphic copy
      this->car = car;
    } else {
      this->car = make_shared<const Tuple>(Tuple{*this});
    }
    if (cdr->typeIndex != typeid(Tuple)) {
      this->cdr = cdr;
    } else {
      this->cdr = make_shared<const Tuple>(Tuple{*this});
    }
  }
};

/**
 * internal value
 */
struct Symbol : private AnyVal {
  string data{};

  explicit Symbol(string data) : data(std::move(data)), AnyVal(typeid(Symbol)) {
  }
};

#endif
