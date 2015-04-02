# where will the python end up in triqs python tree ?
set(python_destination applications/hello_world)

# make a local pytriqs copy to be able to run in the current dir without installation
triqs_prepare_local_pytriqs_merged_with_my_python(${python_destination})

# Build the C++ extension module
triqs_python_extension(hello_world ${python_destination})

# The C++ extension module needs to linked to the C++ lib
target_link_libraries(hello_world hello_world_c)

# The C++ extension module requires some include path, the same as TRIQS plus ./
include_directories(${TRIQS_INCLUDE_ALL} ${CMAKE_CURRENT_SOURCE_DIR})

# This macro sets up the rpath, to avoid any LD_LIBRARY_PATH issue ...
triqs_set_rpath_for_target(hello_world)

# to be able to run from toplevel, we add a link
execute_process(COMMAND ln -fs ${CMAKE_CURRENT_BINARY_DIR}/hello_world.so ${CMAKE_BINARY_DIR}/pytriqs/${python_destination})



