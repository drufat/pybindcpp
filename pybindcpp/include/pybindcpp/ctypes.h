// Copyright (C) 2010-2019 Dzhelil S. Rufat. All Rights Reserved.
#ifndef PYBINDCPP_CTYP_TYPES_H
#define PYBINDCPP_CTYP_TYPES_H

#include <complex>
#include <functional>
#include <iostream>
#include <memory>

#include "api.h"

namespace pybindcpp {

static TypeSystem *ts;

template <typename T> static size_t type_id() {
  static const size_t id = ts->type_counter++;
  return id;
}

template <class T> struct ctype;

template <class T, class... Args> struct sexpr {
  static size_t add(const char *name) {
    const size_t args[] = {ctype<Args>::add()...};
    const size_t nargs = sizeof...(Args);
    auto tid = type_id<T>();
    return ts->add_type(tid, name, args, nargs);
  }
};

template <class T> struct sexpr<T> {
  static size_t add(const char *name) {
    auto tid = type_id<T>();
    return ts->add_type(tid, name, nullptr, 0);
  }
};

template <class T> void deleter(void *ptr) {
  //  std::cerr << "cc del " << reinterpret_cast<void *>(ptr) << std::endl;
  delete static_cast<T *>(ptr);
}

// SIMPLE
template <class T> struct SIMPLE {
  using B = T;
  static B ret(T t) { return t; }
  static T arg(B t) { return t; }
  static Box box(T t) {
    auto ptr = new T(t);
    //    std::cerr << "cc new " << reinterpret_cast<void *>(ptr) << std::endl;
    return {type_id<T>(), ptr, &deleter<T>};
  }
};

template <> struct SIMPLE<void> {};

template <> struct SIMPLE<Box> {
  static Box ret(Box t) { return t; }
  static Box arg(Box t) { return t; }
  static Box box(Box t) { return t; }
};

#define CT(T)                                                                  \
  template <> struct ctype<T> : SIMPLE<T> {                                    \
    static size_t add() { return sexpr<T>::add(#T); }                          \
  };

CT(bool)
CT(wchar_t)
CT(char)
CT(unsigned char)

CT(short)
CT(unsigned short)
CT(int)
CT(unsigned int)
CT(long)
CT(unsigned long)
CT(long long)
CT(unsigned long long)

CT(float)
CT(double)
CT(long double)

CT(char *)
CT(wchar_t *)

CT(void *)
CT(PyObject *)

CT(void)
CT(Box)

CT(std::complex<float>)
CT(std::complex<double>)
CT(std::complex<long double>)

// CONST - ignore for now
template <class T> struct ctype<const T *> : ctype<T *> {
  static Box box(const T *t) { return ctype<T *>::box(const_cast<T *>(t)); }
};

// POINTER
template <class T> struct POINTER {
  using P = T *;
  using B = P;

  static B ret(P p) { return p; }
  static P arg(B b) { return b; }

  static Box box(P p) { return {type_id<P>(), p, nullptr}; }
};

template <class T> struct ctype<T *> : POINTER<T> {
  static size_t add() { return sexpr<T *, T>::add("POINTER"); }
};

// CFUNCTYPE
template <class Ret, class... Args> struct CFUNCTYPE {
  using F = Ret (*)(Args...);
  using B = F;

  static B ret(F f) { return f; }
  static F arg(B b) { return b; }

  static Box box(F f) {
    return {type_id<F>(), reinterpret_cast<void *>(f), nullptr};
  }
};

template <class Ret, class... Args>
struct ctype<Ret (*)(Args...)> : CFUNCTYPE<Ret, Args...> {
  static size_t add() {
    return sexpr<Ret (*)(Args...), Ret, Args...>::add("CFUNCTYPE");
  }
};

struct PyResource {
  void *ptr;
  void (*del)(void *);
  PyResource(void *ptr_, void (*del_)(void *)) : ptr(ptr_), del(del_) {}
  ~PyResource() { del(ptr); }
};

template <class Ret, class... Args> struct Func {
  typename ctype<Ret>::B (*callback)(typename ctype<Args>::B...);
  std::shared_ptr<PyResource> res;

  Func(Box &box) {
    callback = reinterpret_cast<decltype(callback)>(box.ptr);
    res = std::make_shared<PyResource>(box.ptr, box.deleter);
  }
  Ret operator()(Args... args) {
    return ctype<Ret>::arg(callback(ctype<Args>::ret(args)...));
  }
};

template <class... Args> struct Func<void, Args...> {
  void (*callback)(typename ctype<Args>::B...);
  std::shared_ptr<PyResource> res;

  Func(Box &box) {
    callback = reinterpret_cast<decltype(callback)>(box.ptr);
    res = std::make_shared<PyResource>(box.ptr, box.deleter);
  }
  void operator()(Args... args) { callback(ctype<Args>::arg(args)...); }
};

template <class F, class Ret, class... Args> struct Caller {
  using Ret_B = typename ctype<Ret>::B;
  static Ret_B caller(void *ptr, typename ctype<Args>::B... args) {
    F &f = *static_cast<F *>(ptr);
    return ctype<Ret>::ret(f(ctype<Args>::arg(args)...));
  }
};

template <class F, class... Args> struct Caller<F, void, Args...> {
  static void caller(void *ptr, typename ctype<Args>::B... args) {
    F &f = *static_cast<F *>(ptr);
    return f(ctype<Args>::arg(args)...);
  }
};

// CPPFUNCTYPE
template <class F, class Ret, class... Args> struct CPPFUNCTYPE {
  using B = Box;

  static Box box(F f) {
    auto ptr = new F(f);
    //    std::cerr << "cc new " << reinterpret_cast<void *>(ptr) << std::endl;
    return {type_id<F>(), ptr, &deleter<F>};
  }
  static F unbox(Box b) {
    if (b.tid == type_id<F>()) {
      return *static_cast<F *>(b.ptr);
    }
    return Func<Ret, Args...>(b);
  }

  static B ret(F f) { return box(f); }
  static F arg(B b) { return unbox(b); }
};

template <class F, class Ret, class... Args>
struct ctype<Ret (F::*)(Args...) const> : CPPFUNCTYPE<F, Ret, Args...> {
  static size_t add() {
    auto tid = sexpr<F, Ret, Args...>::add("CPPFUNCTYPE");

    using pt = ctype<decltype(Func<Ret, Args...>::callback)>;
    ts->add_callback(tid, pt::add());

    auto call = Caller<F, Ret, Args...>::caller;
    using ct = ctype<decltype(call)>;
    ct::add();
    ts->add_caller(tid, ct::box(call));

    return tid;
  }
};

template <class F> struct ctype : ctype<decltype(&F::operator())> {};

} // namespace pybindcpp

#endif // PYBINDCPP_CTYP_TYPES_H
