# Copy h5 files to binary dir
file(GLOB all_h5_files *.h5)
file(COPY ${all_h5_files} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

set(all_tests anderson_c)

foreach(t ${all_tests})
  add_executable(${t} ${CMAKE_CURRENT_SOURCE_DIR}/${t}.cpp)
  target_link_libraries(${t} ctint_tutorial_c gtest)
  add_test(${t} ${CMAKE_CURRENT_BINARY_DIR}/${t})
endforeach()


