cmake_minimum_required(VERSION 3.6)

set(PYTHON_EXE python3)

execute_process(
        COMMAND ${PYTHON_EXE} -c "import sysconfig as s; print(s.get_path('include'))"
        OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
        COMMAND ${PYTHON_EXE} -c "import numpy as np; print(np.get_include())"
        OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(CMAKE_PYEXT "

import sysconfig
import os
import sys

suffix = sysconfig.get_config_var('EXT_SUFFIX')

def move(name, src):
    src_name, src_ext = os.path.splitext(src)
    assert (src_ext == '.cpp')
    src_dir = os.path.dirname(src_name)
    dest = os.path.join(src_dir, name) + suffix
    return dest


if __name__ == '__main__':
    _, name, src = sys.argv
    print(move(name, src))

")

function(py_module target source)
    add_library(${target} SHARED ${source} ${ARGN})
    target_include_directories(${target} PRIVATE ${PYTHON_INCLUDE_DIR})
    target_include_directories(${target} PRIVATE ${NUMPY_INCLUDE_DIR})
    if (${APPLE})
        set_target_properties(${target} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
    endif ()
    execute_process(
            COMMAND ${PYTHON_EXE} -c "${CMAKE_PYEXT}"
            "${target}" "${CMAKE_SOURCE_DIR}/${source}"
            OUTPUT_VARIABLE OUTPUT
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    add_custom_command(
            OUTPUT ${OUTPUT}
            COMMAND ${CMAKE_COMMAND} -E create_symlink "$<TARGET_FILE:${target}>" "${OUTPUT}"
            DEPENDS ${target}
    )
    add_custom_target("${target}_symlink" ALL DEPENDS ${OUTPUT})
endfunction(py_module)