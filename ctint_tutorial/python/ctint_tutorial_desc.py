# Generated automatically using the command :
# c++2py ../c++/ctint.hpp -p -m triqs.applications.impurity_solvers.ctint_tutorial -o ctint_tutorial -C triqs --cxxflags="-std=c++17"
from cpp2py.wrap_generator import *

# The module
module = module_(full_name = "triqs.applications.impurity_solvers.ctint_tutorial", doc = "", app_name = "triqs.applications.impurity_solvers.ctint_tutorial")

# Imports
import triqs.gf

# Add here all includes
module.add_include("ctint.hpp")

# Add here anything to add in the C++ code at the start, e.g. namespace using
module.add_preamble("""
#include <cpp2py/converters/string.hpp>
#include <triqs/cpp2py_converters/gf.hpp>

""")

module.add_enum("spin", ['spin::up', 'spin::down'], "spin", """""")

# The class ctint_solver
c = class_(
        py_type = "CtintSolver",  # name of the python class
        c_type = "ctint_solver",   # name of the C++ class
        doc = """""",   # doc of the C++ class
        hdf5 = False,
)

c.add_constructor("""(double beta_, int n_iw = 1024, int n_tau = 100001)""", doc = """Construct a ctint solver""")

c.add_method("""void solve (double U, double delta, int n_cycles, int length_cycle = 50, int n_warmup_cycles = 5000, std::string random_name = \"\", int max_time = -1)""",
             doc = """Method that performs the QMC calculation""")

c.add_property(name = "G0_iw",
               getter = cfunction("block_gf_view<triqs::gfs::imfreq> G0_iw ()"),
               doc = """Access non-interacting Matsubara Green function""")

c.add_property(name = "G0_tau",
               getter = cfunction("block_gf_view<triqs::gfs::imtime> G0_tau ()"),
               doc = """Access non-interacting imaginary-time Green function""")

c.add_property(name = "G_iw",
               getter = cfunction("block_gf_view<triqs::gfs::imfreq> G_iw ()"),
               doc = """Access interacting Matsubara Green function""")

module.add_class(c)



module.generate_code()
