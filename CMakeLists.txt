cmake_minimum_required(VERSION 3.6)
project(pybindcpp)

find_package(PythonLibs)

add_compile_options(-std=c++1z)

include_directories(include)
include_directories(/usr/include/python3.5m/)
include_directories(/usr/lib/python3.5/site-packages/numpy/core/include/)

add_custom_target(
        setup_py ALL python3 setup.py develop --user
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

add_custom_target(
        pytest py.test --doctest-modules pybindcpp
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        DEPENDS setup_py
)

add_executable(meta src/meta.cpp)

add_library(bindctypes MODULE src/modules/ext/bindctypes.cpp include/pybindcpp/ctypes.h)

add_library(dispatch MODULE src/modules/ext/dispatch.cpp)
add_library(numpy MODULE src/modules/ext/numpy.cpp)
