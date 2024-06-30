#include <concepts>

namespace concepts_utils {

template <typename T>
concept Comparable = requires(T a, T b) {
  { a < b } -> std::same_as<bool>;
  { a <= b } -> std::same_as<bool>;
};

}; // namespace concepts_utils