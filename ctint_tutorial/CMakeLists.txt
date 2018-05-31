# Version number of the application
set (CTINT_VERSION "1.5")

# Start configuration
cmake_minimum_required(VERSION 2.8)
project(triqs_ctint CXX)

# Default to Release build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Type of build" FORCE)
endif()
message( STATUS "-------- BUILD-TYPE: ${CMAKE_BUILD_TYPE} -------------")

# Use shared libraries
set(BUILD_SHARED_LIBS ON)

# Load TRIQS and Cpp2Py
find_package(TRIQS 2.0 REQUIRED)
find_package(Cpp2Py REQUIRED)

# Default Install directory to TRIQS_ROOT
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT OR (NOT IS_ABSOLUTE ${CMAKE_INSTALL_PREFIX}))
  message(STATUS " No install prefix given (or invalid). Defaulting to TRIQS_ROOT")
  set(CMAKE_INSTALL_PREFIX ${TRIQS_ROOT} CACHE PATH "default install path" FORCE)
endif()
message(STATUS "-------- CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX} -------------")

include_directories(c++)
add_compile_options($<$<CONFIG:Release>:-march=native>
		    $<$<CONFIG:Debug>:-Og>)

triqs_set_rpath_for_all_targets()

# Compile C++ code
add_subdirectory(c++)

# Python interface
add_subdirectory(python)

# Test
enable_testing()
add_subdirectory(test)


