# Version number of the application
set (EXAMPLES_VERSION "1.4")

# Append triqs installed files to the cmake load path
list(APPEND CMAKE_MODULE_PATH ${TRIQS_PATH}/share/triqs/cmake)

# start configuration 
cmake_minimum_required(VERSION 2.8)
project(examples CXX)
set(CMAKE_BUILD_TYPE Release)
enable_testing()

# We use shared libraries
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

# Load TRIQS, including all predefined variables from TRIQS installation
find_package(TRIQS REQUIRED)

# Check that versions are compatible
if(NOT EXAMPLES_VERSION EQUAL TRIQS_VERSION)
  message(FATAL_ERROR "The application version is not compatible with the TRIQS library (TRIQS library version: ${TRIQS_VERSION} while this application version: ${CTINT_VERSION})")
endif()

# We want to be installed in the TRIQS tree
set(CMAKE_INSTALL_PREFIX ${TRIQS_PATH})

add_definitions(${TRIQS_CXX_DEFINITIONS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${TRIQS_INCLUDE_ALL})

if (${TRIQS_USE_CPP14})
set(all_sources array gf gtodelta)
else()
set(all_sources array gtodelta)
MESSAGE(WARNING "gf example only compiles with C++14 mode")
endif()

foreach(t ${all_sources})
add_executable(${t} ${t}.cpp)
target_link_libraries(${t} ${TRIQS_LIBRARY_ALL})
triqs_set_rpath_for_target(${t})
endforeach()
