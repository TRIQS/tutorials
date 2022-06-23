add_library(ctint_tutorial_c ctint.cpp)

target_link_libraries(ctint_tutorial_c PUBLIC triqs)
target_include_directories(ctint_tutorial_c PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/c++>)

install(TARGETS ctint_tutorial_c DESTINATION lib)
