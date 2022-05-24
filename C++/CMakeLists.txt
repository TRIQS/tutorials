# Start configuration
cmake_minimum_required(VERSION 3.12.4)
cmake_policy(VERSION 3.12.4)
project(examples VERSION 1.0 LANGUAGES CXX)

# Default to Release build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Type of build" FORCE)
endif()
message( STATUS "-------- BUILD-TYPE: ${CMAKE_BUILD_TYPE} --------")

# Enable compiler warnings for the whole project
add_definitions(-Wall)

# Load TRIQS, including all predefined variables from TRIQS installation
find_package(TRIQS 3.1 REQUIRED)

set(all_sources array gf gtodelta)
  foreach(t ${all_sources})
  add_executable(${t} ${t}.cpp)
  target_link_libraries(${t} PRIVATE triqs)
endforeach()
