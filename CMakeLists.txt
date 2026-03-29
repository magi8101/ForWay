cmake_minimum_required(VERSION 3.18)
project(forway LANGUAGES C CXX Fortran)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Cross-platform optimization flags
option(FORWAY_PORTABLE "Build portable wheels without -march=native" OFF)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
    if(NOT FORWAY_PORTABLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
    endif()
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math")
    if(NOT FORWAY_PORTABLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    endif()
endif()

# Enable hardware-specific targets for ARM macOS (Apple Silicon) when using GCC/Homebrew
# Apple Silicon always supports fp16, NEON, and other SIMD features.
if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=apple-m1")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -mcpu=apple-m1")
endif()

include(FetchContent)

FetchContent_Declare(
    highway
    GIT_REPOSITORY https://github.com/google/highway.git
    GIT_TAG 1.2.0
    GIT_SHALLOW TRUE
)

FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG v2.4.0
    GIT_SHALLOW TRUE
    GIT_SUBMODULES_RECURSE TRUE
)
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_CONTRIB ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(highway nanobind)

if(CMAKE_VERSION VERSION_LESS 3.18)
    set(DEV_MODULE Development)
else()
    set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.9 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)
find_package(OpenMP COMPONENTS Fortran CXX REQUIRED)

if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -O3 -ffree-form -fno-underscoring -ffast-math -ffree-line-length-none")
    if(NOT FORWAY_PORTABLE)
        set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -march=native")
    endif()
endif()

add_library(micro_kernel STATIC src/micro_kernel.cpp rng/src/rng_micro_kernel.cpp)
target_link_libraries(micro_kernel PRIVATE hwy)
if(NOT MSVC)
    target_compile_options(micro_kernel PRIVATE -O3 -ffast-math $<$<NOT:$<BOOL:${FORWAY_PORTABLE}>>:-march=native>)
endif()
target_include_directories(micro_kernel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/rng/src)

add_library(macro_kernel STATIC src/macro_kernel.f90 rng/src/rng_macro_kernel.f90)
target_link_libraries(macro_kernel PRIVATE OpenMP::OpenMP_Fortran)
target_link_libraries(macro_kernel PRIVATE micro_kernel)

add_library(metrics_kernel STATIC metrics/src/metrics_kernel.cpp metrics/src/dot_kernel.cpp)
target_include_directories(metrics_kernel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/metrics/src)
target_link_libraries(metrics_kernel PRIVATE hwy OpenMP::OpenMP_CXX)

add_library(activations_kernel STATIC activations/src/activations_kernel.cpp)
target_include_directories(activations_kernel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/activations/src)
target_link_libraries(activations_kernel PRIVATE hwy OpenMP::OpenMP_CXX)

add_library(reductions_kernel STATIC reductions/src/reductions_kernel.cpp reductions/src/transpose_kernel.cpp)
target_include_directories(reductions_kernel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/reductions/src)
target_link_libraries(reductions_kernel PRIVATE hwy OpenMP::OpenMP_CXX)

nanobind_add_module(forway NOMINSIZE src/forway.cpp)
target_link_libraries(forway PRIVATE macro_kernel micro_kernel metrics_kernel activations_kernel reductions_kernel hwy hwy_contrib)

install(TARGETS forway LIBRARY DESTINATION .)
