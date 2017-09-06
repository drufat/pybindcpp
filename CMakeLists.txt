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
        pybindcpp/include/pybindcpp/capsule.h
        pybindcpp/include/pybindcpp/module_cpp.h
        pybindcpp/include/pybindcpp/numpy.h
        pybindcpp/include/pybindcpp/storage.h
        pybindcpp/include/pybindcpp/module_cpp_imp.h
        pybindcpp/include/pybindcpp/stringer.h
        pybindcpp/include/pybindcpp/api.h
        pybindcpp/include/pybindcpp/ctypes.h
        pybindcpp/include/pybindcpp/module.h
        pybindcpp/include/pybindcpp/callable_trait.h
        pybindcpp/include/pybindcpp/pyfunction.h
        pybindcpp/include/pybindcpp/func_trait.h
        pybindcpp/include/pybindcpp/cpython_types.h
        pybindcpp/include/pybindcpp/apply.h
        )

#add_executable(main src/main.cpp)

add_custom_target(
        ${PROJECT_NAME}_pytest ALL
        COMMAND py.test -v --doctest-modules pybindcpp
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
function(pymodule target)
    py_module(${target} ${target} ${ARGN} ${HEADER_FILES})
    add_dependencies(${PROJECT_NAME}_pytest ${target} ${target}_symlink)
endfunction()

pymodule(dispatch pybindcpp/dispatch.cpp)

pymodule(bindctypes pybindcpp/ext/bindctypes.cpp)

pymodule(simple pybindcpp/ext/simple.cpp)

pymodule(example pybindcpp/ext/example.cpp)

pymodule(native pybindcpp/ext/native.cpp)

pymodule(native_cpp pybindcpp/ext/native.cpp ${HEADER_FILES})
target_compile_definitions(native_cpp PRIVATE -DNATIVE_CPP)

pymodule(numpy pybindcpp/ext/numpy.cpp)

find_package(PkgConfig REQUIRED)
pkg_check_modules(EIGEN3 eigen3 REQUIRED)
pymodule(eigen pybindcpp/ext/eigen.cpp)
target_include_directories(eigen PRIVATE ${EIGEN3_INCLUDE_DIRS})

pkg_check_modules(FFTW3 fftw3 REQUIRED)
include_directories(${FFTW3_INCLUDE_DIRS})
link_directories(${FFTW3_LIBRARY_DIRS})
pymodule(fftw pybindcpp/ext/fftw.cpp)
target_link_libraries(fftw ${FFTW3_LIBRARIES})

if (DEFINED ENV{AF_PATH})
    include_directories(/usr/local/include)
    link_directories(/usr/local/lib)
    pymodule(arrayfire pybindcpp/ext/arrayfire.cpp)
    target_link_libraries(arrayfire af)
endif ()

add_test(
        NAME PY
        COMMAND py.test -v --doctest-modules pybindcpp
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
