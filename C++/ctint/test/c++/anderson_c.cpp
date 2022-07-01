#include "ctint.hpp"

#include <triqs/test_tools/gfs.hpp>

// Anderson model test
TEST(CtInt, Anderson) {

  // Initialize mpi
  int rank = mpi::communicator().rank();

  // set parameters
  double beta  = 20.0;
  double U     = 1.0;
  double delta = 0.1;
  int n_cycles = 10000;
  int n_iw     = 200;

  // construct the ct-int solver
  auto ctqmc = ctint_solver{beta, n_iw};

  // parameters
  double mu      = 1.3 - U / 2.0; // mu=0 corresponds to half filling
  double epsilon = 0.2;

  // initialize g0(omega)
  nda::clef::placeholder<0> om_;
  for (auto sig : {0, 1}) ctqmc.G0_iw()[sig](om_) << 1.0 / (om_ + mu - 1.0 / (om_ - epsilon));

  // launch the Monte Carlo
  ctqmc.solve(U, delta, n_cycles);

  std::string filename = "anderson_c";
  gf<imfreq, scalar_valued> g_ref;
  if (rank == 0) {
    h5::file f_out(filename + ".out.h5", 'w');
    h5_write(f_out, "G", ctqmc.G_iw()[0]);

    h5::file f_ref(filename + ".ref.h5", 'r');
    h5_read(f_ref, "G", g_ref);
    EXPECT_GF_NEAR(g_ref, ctqmc.G_iw()[0]);
  }
}
MAKE_MAIN;
