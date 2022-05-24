add_library(ctint_tutorial_c ctint.cpp)

target_link_libraries(ctint_tutorial_c PUBLIC triqs)

target_compile_options(ctint_tutorial_c PUBLIC -std=c++17)

# Install
install(TARGETS ctint_tutorial_c DESTINATION lib)
