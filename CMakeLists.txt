cmake_minimum_required(VERSION 3.6)
project(pybindcpp)

add_compile_options(
    -std=c++14
    -Wall
)

find_package(PyBindCpp REQUIRED PATHS cmake)
include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${NUMPY_INCLUDE_DIR})

include_directories(pybindcpp/include)

set(HEADER_FILES
    pybindcpp/include/capi/module.h
    pybindcpp/include/capi/module_imp.h
    pybindcpp/include/capi/capsule.h
    pybindcpp/include/capi/cpython_types.h
    pybindcpp/include/capi/apply.h
    pybindcpp/include/capi/stringer.h
    pybindcpp/include/ctyp/api.h
    pybindcpp/include/ctyp/ctypes.h
    pybindcpp/include/ctyp/module.h
    pybindcpp/include/ctyp/callable_trait.h
    pybindcpp/include/ctyp/pyfunction.h
    pybindcpp/include/ctyp/func_trait.h
    pybindcpp/include/numpy_pybind.h
 )

#add_custom_target(
#    ${PROJECT_NAME}_pytest ALL
#    COMMAND py.test -v --doctest-modules pybindcpp
#    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#)

function(pymodule target)
    py_module(${target} ${target} ${ARGN} ${HEADER_FILES})
#    add_dependencies(${PROJECT_NAME}_pytest ${target} ${target}_symlink)
endfunction()

pymodule(simple pybindcpp/ext/simple.cpp)

#pymodule(dispatch pybindcpp/dispatch.cpp)

#pymodule(bindctypes pybindcpp/ext/bindctypes.cpp)

#pymodule(example pybindcpp/ext/example.cpp)

#pymodule(native pybindcpp/ext/native.cpp)

#pymodule(native_cpp pybindcpp/ext/native.cpp)
#target_compile_definitions(native_cpp PRIVATE -DNATIVE_CPP)

#pymodule(numpy pybindcpp/ext/numpy/numpy.cpp)

#find_package(PkgConfig REQUIRED)
#pkg_check_modules(EIGEN3 eigen3 REQUIRED)
#pymodule(eigen pybindcpp/ext/numpy/eigen.cpp)
#target_include_directories(eigen PRIVATE ${EIGEN3_INCLUDE_DIRS})

#pkg_check_modules(FFTW3 fftw3 REQUIRED)
#include_directories(${FFTW3_INCLUDE_DIRS})
#link_directories(${FFTW3_LIBRARY_DIRS})
#pymodule(fftw pybindcpp/ext/numpy/fftw.cpp)
#target_link_libraries(fftw ${FFTW3_LIBRARIES})
