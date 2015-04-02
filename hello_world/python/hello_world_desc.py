
# Generated automatically using the command : 
# wrapper_desc_generator.py ../c++/hello_world.hpp -p -o hello_world
from wrap_generator import *

# The module
module = module_(full_name = "hello_world", doc = "")

# All the triqs C++/Python modules

# Add here all includes beyond what is automatically included by the triqs modules
module.add_include("../c++/hello_world.hpp")

# Add here anything to add in the C++ code at the start, e.g. namespace using
module.add_preamble("""
""")
module.add_function ("std::string hello ()", doc = "")

module.generate_code()

