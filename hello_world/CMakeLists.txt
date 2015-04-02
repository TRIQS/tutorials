# start configuration 
cmake_minimum_required(VERSION 2.8)
project(hello_world CXX)
set(CMAKE_BUILD_TYPE Release)
enable_testing()

# We use shared libraries
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

# Load TRIQS, including all predefined variables from TRIQS installation
# First append triqs installed files to the cmake load path
list(APPEND CMAKE_MODULE_PATH ${TRIQS_PATH}/share/triqs/cmake)
find_package(TRIQS REQUIRED)

# We want to be installed in the TRIQS tree
#set(CMAKE_INSTALL_PREFIX ${TRIQS_PATH})
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/INSTALL_DIR)

# Compile C++ code
add_subdirectory(c++)

# Python interface (if is compiled with Python support).
# Remove this for a pure C++ code.
if (${TRIQS_WITH_PYTHON_SUPPORT})
 add_subdirectory(python)
endif()

# Add tests
#add_subdirectory(test)

