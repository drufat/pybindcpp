cmake_minimum_required(VERSION 3.6)
project(pybindcpp)

add_compile_options(-std=c++1z)

add_custom_target(
        pythonmodule ALL python3 setup.py develop --user
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

add_custom_target(
        pytest py.test --doctest-modules pybindcpp
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

add_executable(meta src/meta.cpp)
