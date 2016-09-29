cmake_minimum_required(VERSION 3.6)
project(pybindcpp)

find_package(PythonLibs)

add_compile_options(-std=c++1z)

# Linux
include_directories(include)
include_directories(/usr/include/python3.5m/)
include_directories(/usr/lib/python3.5/site-packages/numpy/core/include/)

# Mac OS X
include_directories(/usr/local/Cellar/python3/3.5.2_1/Frameworks/Python.framework/Versions/3.5/include/python3.5m/)
include_directories(/usr/local/lib/python3.5/site-packages/numpy/core/include/)
link_directories(/usr/local/Cellar/python3/3.5.2_1/Frameworks/Python.framework/Versions/3.5/lib/)

set(HEADER_FILES
        include/pybindcpp/capsule.h
        include/pybindcpp/module.h
        include/pybindcpp/numpy.h
        include/pybindcpp/storage.h
        include/pybindcpp/types.h
        include/pybindcpp/util.h
        include/pybindcpp/ctypes/ctypes.h
        #        include/pybindcpp/ctypes/capsule.h
        #        include/pybindcpp/ctypes/types.h
        #        include/pybindcpp/ctypes/module.h
        #        include/pybindcpp/ctypes/callable.h
        )

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

set(CMAKE_CXX_STANDARD_LIBRARIES -lpython3.5m)
add_library(bindctypes MODULE src/modules/ext/bindctypes.cpp ${HEADER_FILES})
add_library(dispatch MODULE src/modules/ext/dispatch.cpp ${HEADER_FILES})
add_library(numpy MODULE src/modules/ext/numpy.cpp ${HEADER_FILES})
