# Build the python module
add_cpp2py_module(ctint_tutorial)

target_link_libraries(ctint_tutorial ctint_tutorial_c)

set(PYTHON_SOURCES
 ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
 )

# Install python module to proper location
set(PYTHON_LIB_DEST ${CPP2PY_PYTHON_LIB_DEST_ROOT}/ctint_tutorial)
install(TARGETS ctint_tutorial DESTINATION ${PYTHON_LIB_DEST})
install(FILES ${PYTHON_SOURCES} DESTINATION ${PYTHON_LIB_DEST})

