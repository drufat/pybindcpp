cmake_minimum_required(VERSION 3.6)
project(pybindcpp)

add_compile_options(-std=c++14)

include_directories(include)

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
set(CMAKE_CXX_STANDARD_LIBRARIES ${PYTHON_LIBRARY})

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

add_executable(main main.cpp)

function(py_module target source)
    python_add_module(${target} ${source} ${HEADER_FILES})
    add_dependencies(pytest ${target})
    add_custom_command(
            TARGET ${target}
            POST_BUILD
            COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/cmake_pyext.py"
            "${target}" "$<TARGET_FILE:${target}>" "${CMAKE_SOURCE_DIR}/${source}"
    )
endfunction(py_module)

add_custom_target(
        pytest ALL py.test -v --doctest-modules ${CMAKE_PROJECT_NAME}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/..
)

py_module(dispatch ../pybindcpp/dispatch.cpp)

py_module(bindctypes ../pybindcpp/ext/bindctypes.cpp)

py_module(simple ../pybindcpp/ext/simple.cpp)

py_module(native ../pybindcpp/ext/native.cpp)

py_module(native_cpp ../pybindcpp/ext/native.cpp)
target_compile_definitions(native_cpp PRIVATE -DNATIVE_CPP)

py_module(numpy ../pybindcpp/ext/numpy.cpp)

find_package(PkgConfig REQUIRED)
pkg_check_modules(EIGEN3 eigen3)
py_module(eigen ../pybindcpp/ext/eigen.cpp)

include_directories(${EIGEN3_INCLUDE_DIRS})
pkg_check_modules(FFTW3 fftw3)
py_module(fftw ../pybindcpp/ext/fftw.cpp)
target_link_libraries(fftw ${FFTW3_LIBRARIES})

py_module(arrayfire ../pybindcpp/ext/arrayfire.cpp)
target_link_libraries(arrayfire af)
