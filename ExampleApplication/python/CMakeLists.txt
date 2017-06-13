# Where will the python end up in triqs python tree ?
set(python_destination pytriqs/applications/hello_world)

# Make a local pytriqs copy to be able to run in the current dir without installation
triqs_prepare_local_pytriqs(${python_destination})

# Build the C++ extension module
triqs_python_extension(hello_world ${python_destination})

# The C++ extension module needs to linked to the C++ lib
target_link_libraries(hello_world hello_world_c)

# The C++ extension module requires some include path, the same as TRIQS plus ./
include_directories(${TRIQS_INCLUDE_ALL} ${CMAKE_CURRENT_SOURCE_DIR})

# This macro sets up the rpath, to avoid any LD_LIBRARY_PATH issue ...
triqs_set_rpath_for_target(hello_world)

# Install python module into the triqs folder structure
install(TARGETS hello_world DESTINATION ${TRIQS_PYTHON_LIB_DEST_ROOT}/${python_destination})
