cmake_minimum_required(VERSION 3.24)

# Must run before project(): the toolchain governs dependency resolution.
include(${CMAKE_CURRENT_LIST_DIR}/cmake/vcpkg.cmake)

option(IE_ENABLE_CUDA "Build the InferenceEngine CUDA backend" ON)

project(ModelGraph LANGUAGES CXX)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type for single-config generators" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# All tensor maths, layers, tokenizers and checkpoint loading live in the
# InferenceEngine repository; this project only adds the graph layer on top.
# By default its latest GitHub release is downloaded (see cmake/InferenceEngine.cmake).
include(cmake/InferenceEngine.cmake)

if(MSVC)
    set(MG_WARNINGS /W4 /permissive- /utf-8 /Zc:preprocessor /bigobj)
else()
    set(MG_WARNINGS -Wall -Wextra)
endif()

add_subdirectory(src)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
