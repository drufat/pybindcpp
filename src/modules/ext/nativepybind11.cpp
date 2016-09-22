// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#include <functional>
#include <pybind11/pybind11.h>
#include <python/capsule.h>

namespace py = pybind11;

struct MyClass {

    int memberdata;

    MyClass(int a) {
        memberdata = a;
    }

    int get() {
        return memberdata;
    }

    void increment() {
        memberdata++;
    }

    void set(int a) {
        memberdata = a;
    }
};

int
f(int N, int n, int x)
{
    return N + n + x;
}

int
g(int x, int y) {
    return x  + y;
}

PYBIND11_PLUGIN(nativepybind11) {
    py::module m("nativepybind11");

    m.attr("one") = py::int_(1);
    m.attr("two") = py::int_(2);
    m.attr("true") = py::bool_(true);
    m.attr("false")= py::bool_(false);
    m.attr("name") = py::str("native");
    m.attr("name1") = py::str(std::string("native"));

    m.def("func", &f);

    static int N, n, x;

    m.def("f", [](int N_, int n_, int x_){
        N = N_;
        n = n_;
        x = x_;
        return f(N, n, x);
    });

    m.def("closure", [](){
        return std::tuple<int, int, int>(N, n, x);
    });

    m.def("S", [](py::object o){
        return o;
    });

    m.def("g_cfun", g);
    m.def("g_fun", std::function<int(int,int)>(g));
    m.def("g_afun", [](int x, int y) -> int {
        return g(x, y);
    });
    m.attr("g_ofun") = py::cpp_function(g);

    auto f_one = std::function<int()>([]() {
        return 1;
    });
    m.def("f_one", f_one);

    py::object f_one_py = m.attr("f_one");

    m.def("f_func", [=](){
        return f_one_py;
    });

    m.attr("caps_int") = python::capsule_new(std::make_shared<int>(3));
    m.attr("caps_double") = python::capsule_new(std::make_shared<double>(3.0));
    m.attr("caps_string") = python::capsule_new(std::make_shared<std::string>("Hello!"));
    m.def("PyCapsule_GetName", [](py::object o){
        return PyCapsule_GetName(o.ptr());
    });

    pybind11::class_<MyClass>(m, "MyClass")
            .def(pybind11::init<int>())
            .def("increment", &MyClass::increment)
            .def("set", &MyClass::set)
            .def("get", &MyClass::get);

    return m.ptr();
}
