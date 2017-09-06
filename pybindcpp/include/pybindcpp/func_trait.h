#ifndef PYBINDCPP_FUNC_TRAIT_H
#define PYBINDCPP_FUNC_TRAIT_H

#include <array>
#include <sstream>

#include "pybindcpp/api.h"
#include "pybindcpp/ctypes.h"

namespace pybindcpp {

template <class F>
struct func_trait;

template <class Ret, class... Args>
struct func_trait<Ret (*)(Args...)>
{

  static constexpr size_t size = 1 + sizeof...(Args);

  static auto value()
  {
    const std::array<std::type_index, size> a = {
      { typeid(typename std::decay<Ret>::type),
        typeid(typename std::decay<Args>::type)... }
    };
    return a;
  }

  static auto str()
  {
    auto val = value();
    std::stringstream ss;
    ss << "(";
    for (size_t i = 0; i < size; i++) {
      ss << ctype_map.at(val[i]) << ",";
    }
    ss << ")";
    return ss.str();
  }

  static auto pystr() { return PyBytes_FromString(str().c_str()); }

  static auto pyctype()
  {
    auto s = str();
    return api->get_type(s.c_str());
  }
};
}

#endif // PYBINDCPP_FUNC_TRAIT_H
