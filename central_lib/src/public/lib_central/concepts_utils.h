#ifndef DEV_QC_CENTRAL_LIB_CONCEPTS_UTILS
#define DEV_QC_CENTRAL_LIB_CONCEPTS_UTILS

#include "workspace_pch.h"

namespace concepts_utils {

template <typename T>
concept Comparable = requires(T a, T b) {
    { a < b } -> std::same_as<bool>;
    { a <= b } -> std::same_as<bool>;
};

}; // namespace concepts_utils

#endif // DEV_QC_CENTRAL_LIB_CONCEPTS_UTILS