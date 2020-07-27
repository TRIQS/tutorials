#include <triqs/gfs.hpp>
using namespace triqs::gfs;
using namespace triqs::lattice;
int main() {

 double beta = 10, mu = 0;
 int n_freq = 100, n_pts = 100;

 // Green's function on Matsubara frequencies, 1x1 matrix-valued.
 auto Delta_iw = gf<imfreq>{{beta, Fermion, n_freq}, {1, 1}};
 auto Gloc = gf<imfreq>{{beta, Fermion, n_freq}, {1, 1}};

 // Green's function in imaginary time, 1x1 matrix-valued.
 auto Delta_tau = gf<imtime>{{beta, Fermion, 2 * n_freq + 1}, {1, 1}};

 auto bz = brillouin_zone{bravais_lattice{{{1, 0}, {0, 1}}}};
 auto bz_mesh = gf_mesh<brillouin_zone>{bz, n_pts};

 triqs::clef::placeholder<1> k_;
 triqs::clef::placeholder<2> iw_;

 // The actual equations
 Gloc(iw_) << sum(1 / (iw_ + mu - 2 * (cos(k_[0]) + cos(k_[1]))), k_ = bz_mesh) / bz_mesh.size(); // (3)
 Delta_iw(iw_) << iw_ + mu - 1 / Gloc(iw_);                                                       // (4)
 Delta_tau() = fourier(Delta_iw);                                                         	  // (5)

 // Write the hybridization to an HDF5 archive
 auto file = h5::file("Delta.h5", 'w');
 h5_write(file, "Delta_tau", Delta_tau);
 h5_write(file, "Delta_iw", Delta_iw);
}
