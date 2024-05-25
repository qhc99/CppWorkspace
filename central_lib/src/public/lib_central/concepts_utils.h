#include <concepts>

namespace concepts_utils {
template <typename T>
concept Assignable = requires(T a, T b) {
  { a = b } -> std::same_as<T &>;
};
}; // namespace concepts_utils