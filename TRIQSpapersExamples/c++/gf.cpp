#include <triqs/gfs.hpp>
using namespace triqs;
using namespace triqs::gfs;
using namespace triqs::lattice;
int main() {

 double beta = 10;
 int n_freq = 1000;

 clef::placeholder<0> iw_;
 clef::placeholder<1> k_;

 // Construction of a 1x1 matrix-valued fermionic gf on Matsubara frequencies.
 auto g_iw = gf<imfreq>{{beta, Fermion, n_freq}, {1, 1}};

 // Automatic placeholder evaluation
 g_iw(iw_) << 1 / (iw_ + 2);

 // Inverse Fourier transform to imaginary time
 auto g_tau = gf<imtime>{{beta, Fermion, 2 * n_freq + 1}, {1, 1}};
 g_tau() = inverse_fourier(g_iw); // Fills a full view of g_tau with FFT result

 // Create a block Green's function composed of three blocks,
 // labeled a,b,c and made of copies of the g_iw functions.
 auto G_iw = make_block_gf({"a", "b", "c"}, {g_iw, g_iw, g_iw});

 // A multivariable gf: G(k,omega)
 auto bz = brillouin_zone{bravais_lattice{{{1, 0}, {0, 1}}}};
 auto g_k_iw = gf<cartesian_product<brillouin_zone, imfreq>>{
     {{bz, 100}, {beta, Fermion, n_freq}}, {1, 1}};

 g_k_iw(k_, iw_) << 1 / (iw_ - 2 * (cos(k_(0)) + cos(k_(1))) - 1 / (iw_ + 2));

 // Writing the Green's functions into an HDF5 file.
 auto f = h5::file("file_g_k_iw.h5", H5F_ACC_TRUNC);
 h5_write(f, "g_k_iw", g_k_iw);
 h5_write(f, "g_iw", g_iw);
 h5_write(f, "g_tau", g_tau);
 h5_write(f, "block_gf", G_iw);
}

