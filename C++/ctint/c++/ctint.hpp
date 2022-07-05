#include <triqs/gfs.hpp>
#include <triqs/mesh.hpp>

// ------------ The main class of the solver -----------------------

using namespace triqs::gfs;
using namespace nda;

enum spin { up, down };

class ctint_solver {
  double beta;
  int n_matsubara, n_times_slices;

  public:
  block_gf<imfreq, scalar_valued> G0_iw, G0tilde_iw, G_iw, M_iw;
  block_gf<imtime, scalar_valued> G0tilde_tau, M_tau;
  nda::array<dcomplex, 1> M_hatree = nda::zeros<dcomplex>(2);

  /// Construct a ctint solver
  ctint_solver(double beta_, int n_iw = 1024, int n_tau = 100001);

  /// Method that performs the QMC calculation
  void solve(double U, double delta, int n_cycles, int length_cycle = 50, int n_warmup_cycles = 5000, std::string random_name = "",
             int max_time = -1);
};
