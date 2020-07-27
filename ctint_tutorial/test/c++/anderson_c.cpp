#include "ctint.hpp"
#include <triqs/gfs.hpp>
#include <triqs/test_tools/gfs.hpp>

// Anderson model test
TEST(CtInt, Anderson) {

  // Initialize mpi
  int rank = triqs::mpi::communicator().rank();

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
  triqs::clef::placeholder<0> om_;
  for (auto sig : {0, 1}) ctqmc.G0_iw()[sig](om_) << 1.0 / (om_ + mu - 1.0 / (om_ - epsilon));

  // launch the Monte Carlo
  ctqmc.solve(U, delta, n_cycles);

  // to compare with ct_seg
  // gf<imfreq> gw = ctqmc.G0_iw()[0];
  // auto gt = make_gf_from_inverse_fourier(gw);

  std::string filename = "anderson_c";
  gf<imfreq> g;
  if (rank == 0) {
    h5::file G_file(filename + ".ref.h5", 'r');
    h5_read(G_file, "G", g);
    EXPECT_GF_NEAR(g, ctqmc.G_iw()[0]);
  }
  if (rank == 0) {
    h5::file G_file(filename + ".out.h5", 'w');
    h5_write(G_file, "G", ctqmc.G_iw()[0]);
  }
}
MAKE_MAIN;
