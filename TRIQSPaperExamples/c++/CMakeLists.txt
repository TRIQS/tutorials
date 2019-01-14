# Version number of the application
set(EXAMPLES_VERSION 2.0)

# Start configuration
cmake_minimum_required(VERSION 2.8.12)
project(app4triqs CXX)

# Default to Release build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Type of build" FORCE)
endif()
message( STATUS "-------- BUILD-TYPE: ${CMAKE_BUILD_TYPE} --------")

# Enable compiler warnings for the whole project
add_definitions(-Wall)

# Load TRIQS, including all predefined variables from TRIQS installation
find_package(TRIQS 2.0 REQUIRED)
find_package(Cpp2Py REQUIRED)

# Default Install directory to TRIQS_ROOT if not given or invalid.
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT OR (NOT IS_ABSOLUTE ${CMAKE_INSTALL_PREFIX}))
  message(STATUS "No install prefix given (or invalid). Defaulting to TRIQS_ROOT")
  set(CMAKE_INSTALL_PREFIX ${TRIQS_ROOT} CACHE PATH "default install path" FORCE)
endif()
message(STATUS "-------- INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX} --------")

set(all_sources array gf gtodelta)
foreach(t ${all_sources})
add_executable(${t} ${t}.cpp)
target_link_libraries(${t} triqs)
target_compile_options(${t} PRIVATE -std=c++17)
endforeach()
