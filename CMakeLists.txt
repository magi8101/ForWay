cmake_minimum_required(VERSION 3.18)
project(forway LANGUAGES C CXX Fortran)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
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
set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(highway nanobind)

if(CMAKE_VERSION VERSION_LESS 3.18)
    set(DEV_MODULE Development)
else()
    set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.9 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)
find_package(OpenMP COMPONENTS Fortran REQUIRED)

set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -O3")
set(CMAKE_Fortran_FLAGS_RELEASE "-O3")

if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -ffree-form -fno-underscoring -O3 -ffast-math -march=native")
endif()

add_library(micro_kernel STATIC src/micro_kernel.cpp rng/src/rng_micro_kernel.cpp)
target_link_libraries(micro_kernel PRIVATE hwy)
target_compile_options(micro_kernel PRIVATE -O3 -ffast-math -march=native)
target_include_directories(micro_kernel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/rng/src)

add_library(macro_kernel STATIC src/macro_kernel.f90 rng/src/rng_macro_kernel.f90)
target_link_libraries(macro_kernel PRIVATE OpenMP::OpenMP_Fortran)
target_link_libraries(macro_kernel PRIVATE micro_kernel)

nanobind_add_module(forway NOMINSIZE src/forway.cpp)
target_link_libraries(forway PRIVATE macro_kernel micro_kernel hwy)

install(TARGETS forway LIBRARY DESTINATION .)
