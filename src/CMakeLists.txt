cmake_minimum_required(VERSION 3.6)
project(pybindcpp)

add_compile_options(-std=c++14)

set(PYTHON_EXE python3)

include_directories(include)
include_directories(/usr/local/include)
link_directories(/usr/local/lib)
if (${APPLE})
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -undefined dynamic_lookup")
endif ()

execute_process(
        COMMAND ${PYTHON_EXE} -c "import sysconfig as s; print(s.get_path('include'))"
        OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
include_directories(${PYTHON_INCLUDE_DIR})

execute_process(
        COMMAND ${PYTHON_EXE} -c "import numpy as np; print(np.get_include())"
        OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
include_directories(${NUMPY_INCLUDE_DIR})

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
    add_library(${target} SHARED ${source} ${HEADER_FILES})
    add_dependencies(pytest ${target})
    add_custom_command(
            TARGET ${target}
            POST_BUILD
            COMMAND ${PYTHON_EXE} "${CMAKE_CURRENT_SOURCE_DIR}/cmake_pyext.py"
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
target_include_directories(eigen PRIVATE ${EIGEN3_INCLUDE_DIRS})

pkg_check_modules(FFTW3 fftw3)
include_directories(${FFTW3_INCLUDE_DIRS})
link_directories(${FFTW3_LIBRARY_DIRS})
py_module(fftw ../pybindcpp/ext/fftw.cpp)
target_link_libraries(fftw ${FFTW3_LIBRARIES})

py_module(arrayfire ../pybindcpp/ext/arrayfire.cpp)
target_link_libraries(arrayfire af)
