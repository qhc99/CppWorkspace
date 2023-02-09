#ifndef QC_CPP_PROJECTS_LISP_UTILS_H
#define QC_CPP_PROJECTS_LISP_UTILS_H
#include "exceptions.h"
#include "values.h"
#include <string>

inline void require(std::shared_ptr<Value> x, bool predicate, std::string m) {
  if (!predicate) {
    throw SyntaxException{(x->to_string() + " " + m).c_str()};
  }
}
inline void require(std::shared_ptr<Value> x, bool predicate) {
  require(x, predicate, "wrong length");
}

#endif