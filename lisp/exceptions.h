//
// Created by Nathan on 2023-02-01.
//

#ifndef DEV_QHC_CPP_PROJECTS_EXCEPTIONS_H
#define DEV_QHC_CPP_PROJECTS_EXCEPTIONS_H

#include <exception>
#include <string>

using std::string;

class BaseException : public std::exception {
  string msg;

public:
  /**
   *
   * @param msg will be copied
   * @param type
   */
  inline explicit BaseException(const char *msg, const std::type_info &type) {
    string a{type.name()};
    a = "[" + a;
    a += "]: ";
    string b{msg};
    this->msg = a + b;
  }

  [[nodiscard]] inline const char *what() const noexcept override {
    return msg.c_str();
  }
};

class SyntaxException : BaseException {
public:
  inline explicit SyntaxException(const char *msg)
      : BaseException(msg, typeid(SyntaxException)) {}
};

class ArgumentsCountException : BaseException {
public:
  inline explicit ArgumentsCountException(const char *msg)
      : BaseException(msg, typeid(ArgumentsCountException)) {}
};

class LookupException : BaseException {
public:
  inline explicit LookupException(const char *msg)
      : BaseException(msg, typeid(LookupException)) {}
};

class TypeException : BaseException {
public:
  inline explicit TypeException(const char *msg)
      : BaseException(msg, typeid(TypeException)) {}
};

#endif // DEV_QHC_CPP_PROJECTS_EXCEPTIONS_H
