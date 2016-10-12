cmake_minimum_required(VERSION 3.6)
project(pybindcpp)

add_compile_options(-std=c++14)

include_directories(include)

# Linux
include_directories(/usr/include/python3.5m/)
include_directories(/usr/lib/python3.5/site-packages/numpy/core/include/)

# Mac OS X
include_directories(/usr/local/Cellar/python3/3.5.2_2/Frameworks/Python.framework/Versions/3.5/include/python3.5m/)
include_directories(/usr/local/lib/python3.5/site-packages/numpy/core/include/)
link_directories(/usr/local/Cellar/python3/3.5.2_2/Frameworks/Python.framework/Versions/3.5/lib/)

set(HEADER_FILES
        include/pybindcpp/capsule.h
        include/pybindcpp/module_cpp.h
        include/pybindcpp/numpy.h
        include/pybindcpp/storage.h
        include/pybindcpp/module_cpp_imp.h
        include/pybindcpp/stringer.h
        include/pybindcpp/api.h
        include/pybindcpp/ctypes.h
        include/pybindcpp/module.h
        include/pybindcpp/callable_trait.h
        include/pybindcpp/pyfunction.h
        include/pybindcpp/func_trait.h
        include/pybindcpp/cpython_types.h
        include/pybindcpp/apply.h
        )

add_custom_target(
        setup_py ALL python3 setup.py develop --user
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/..
)

add_custom_target(
        pytest ALL py.test -vv --doctest-modules pybindcpp
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/..
        DEPENDS setup_py
)

add_executable(main main.cpp)

set(CMAKE_CXX_STANDARD_LIBRARIES -lpython3.5m)
add_library(bindctypes MODULE ../pybindcpp/ext/bindctypes.cpp ${HEADER_FILES})
add_library(numpy MODULE ../pybindcpp/ext/numpy.cpp ${HEADER_FILES})
#add_library(eigen MODULE ../pybindcpp/ext/eigen.cpp ${HEADER_FILES})
#add_library(fftw MODULE ../pybindcpp/ext/fftw.cpp ${HEADER_FILES})
add_library(native MODULE ../pybindcpp/ext/native.cpp ${HEADER_FILES})
