# CMake Start configuration
cmake_minimum_required(VERSION 3.12.4 FATAL_ERROR)
cmake_policy(VERSION 3.12.4)
if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

# ############
# Define Project
project(ctint_tutorial VERSION 3.1.0 LANGUAGES C CXX)

# Default to Release build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Type of build" FORCE)
endif()
message( STATUS "-------- BUILD-TYPE: ${CMAKE_BUILD_TYPE} -------------")

# ############
# Load TRIQS and CPP2PY
find_package(TRIQS 3.1 REQUIRED)

# Default Install directory to TRIQS_ROOT if not given or invalid.
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT OR (NOT IS_ABSOLUTE ${CMAKE_INSTALL_PREFIX}))
  message(STATUS "No install prefix given (or invalid). Defaulting to TRIQS_ROOT")
  set(CMAKE_INSTALL_PREFIX ${TRIQS_ROOT} CACHE PATH "default install path" FORCE)
  set(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT FALSE)
endif()
message(STATUS "-------- CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX} --------")

# Default to Release build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Type of build" FORCE)
endif()
if(NOT IS_SUBPROJECT)
  message(STATUS "-------- BUILD-TYPE: ${CMAKE_BUILD_TYPE} --------")
endif()

# Export the list of compile-commands into compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Testing
option(Build_Tests "Build tests" ON)
if(Build_Tests)
  enable_testing()
endif()

# #############
# Build Project
#
# Build and install the library
add_subdirectory(c++)

# Python interface
add_subdirectory(python)

# Tests
if(Build_Tests)
  add_subdirectory(test)
endif()
