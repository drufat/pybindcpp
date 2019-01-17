#ifndef PYBINDCPP_FUNC_TRAIT_H
#define PYBINDCPP_FUNC_TRAIT_H

#include <array>
#include <sstream>

#include "pybindcpp/api.h"
#include "pybindcpp/ctypes.h"

namespace pybindcpp {

template <class F> struct func_trait;

template <class Ret, class... Args> struct func_trait<Ret (*)(Args...)> {

  static constexpr size_t size = 1 + sizeof...(Args);

  static auto str() {

    static const size_t signature[size] = {typeid(Ret).hash_code(),
                                           typeid(Args).hash_code()...};

    std::stringstream ss;
    ss << "(";
    for (size_t i = 0; i < size; i++) {
      ss << ctype_map.at(signature[i]) << ",";
    }
    ss << ")";

    return ss.str();
  }
}; // namespace pybindcpp
} // namespace pybindcpp

#endif // PYBINDCPP_FUNC_TRAIT_H
