# Generated automatically using the command :
# c++2py ../c++/ctint.hpp -p --members_read_only -a ctint_tutorial -m ctint_tutorial -o ctint_tutorial -C triqs --moduledoc="CTInt Tutorial" --includes=../c++ --cxxflags="-std=c++20 $(triqs++ -cxxflags)" --target_file_only
from cpp2py.wrap_generator import *

# The module
module = module_(full_name = "ctint_tutorial", doc = r"CTInt Tutorial", app_name = "ctint_tutorial")

# Imports
module.add_imports(*['triqs.gf', 'triqs.gf.meshes'])

# Add here all includes
module.add_include("ctint.hpp")

# Add here anything to add in the C++ code at the start, e.g. namespace using
module.add_preamble("""
#include <cpp2py/converters/string.hpp>
#include <triqs/cpp2py_converters/gf.hpp>

""")

module.add_enum("spin", ['spin::up', 'spin::down'], "spin", doc = r"""""")

# The class ctint_solver
c = class_(
        py_type = "CtintSolver",  # name of the python class
        c_type = "ctint_solver",   # name of the C++ class
        doc = r"""""",   # doc of the C++ class
        hdf5 = False,
)

c.add_constructor("""(double beta_, int n_iw = 1024, int n_tau = 100001)""", doc = r"""Construct a ctint solver""")

c.add_method("""void solve (double U, double delta, int n_cycles, int length_cycle = 50, int n_warmup_cycles = 5000, std::string random_name = \"\", int max_time = -1)""",
             doc = r"""Method that performs the QMC calculation""")

c.add_property(name = "G0_iw",
               getter = cfunction("block_gf_view<triqs::mesh::imfreq> G0_iw ()"),
               doc = r"""Access non-interacting Matsubara Green function""")

c.add_property(name = "G0_tau",
               getter = cfunction("block_gf_view<triqs::mesh::imtime> G0_tau ()"),
               doc = r"""Access non-interacting imaginary-time Green function""")

c.add_property(name = "G_iw",
               getter = cfunction("block_gf_view<triqs::mesh::imfreq> G_iw ()"),
               doc = r"""Access interacting Matsubara Green function""")

module.add_class(c)



module.generate_code()