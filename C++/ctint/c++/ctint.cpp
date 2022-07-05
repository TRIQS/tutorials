#include "ctint.hpp"

#include <mpi/mpi.hpp>

#include <triqs/mc_tools.hpp>
#include <triqs/det_manip.hpp>

// --------------- The QMC configuration ----------------

// Operator types
struct c_t {
  double tau; // Imaginary time
  int s;      // Auxiliary spin
};
struct cdag_t {
  double tau;
  int s;
};

// The function that appears in the calculation of the determinant
struct g0bar_tau {
  gf<imtime, scalar_valued> const &gt;
  double beta, delta;
  int s;

  dcomplex operator()(c_t const &c, cdag_t const &cdag) const {
    if ((c.tau == cdag.tau)) { // G_\sigma(0^-) - \alpha(\sigma s)
      return 1.0 + gt[0] - (0.5 + (s == c.s ? 1 : -1) * delta);
    }
    auto tau = c.tau - cdag.tau;
    if (tau >= 0)
      return gt[closest_mesh_pt(tau)];
    else // tau < 0, Account for anti-periodicity
      return -gt[closest_mesh_pt(tau + beta)];
  }
};

// The Monte Carlo configuration
struct configuration {
  // M-matrices for up and down
  std::vector<triqs::det_manip::det_manip<g0bar_tau>> Mmatrices;

  int perturbation_order() const { return Mmatrices[up].size(); }

  configuration(block_gf<imtime, scalar_valued> &G0tilde_tau, double beta, double delta) {
    // Initialize the M-matrices. 100 is the initial matrix size
    for (auto spin : {up, down}) Mmatrices.emplace_back(g0bar_tau{G0tilde_tau[spin], beta, delta, spin}, 100);
  }
};

// ------------ QMC move : inserting a vertex ------------------

struct move_insert {
  configuration *config;
  triqs::mc_tools::random_generator &rng;
  double beta, U;

  dcomplex attempt() { // Insert an interaction vertex at time tau with aux spin s
    double tau     = rng(beta);
    int s          = rng(2);
    auto k         = config->perturbation_order();
    auto det_ratio = config->Mmatrices[up].try_insert(k, k, {tau, s}, {tau, s}) * config->Mmatrices[down].try_insert(k, k, {tau, s}, {tau, s});
    return -beta * U / (k + 1) * det_ratio; // The Metropolis ratio
  }

  dcomplex accept() {
    for (auto &d : config->Mmatrices) d.complete_operation(); // Finish insertion
    return 1.0;
  }

  void reject() {
    for (auto &d : config->Mmatrices) d.reject_last_try(); // Finish insertion
  }
};

// ------------ QMC move : deleting a vertex ------------------

struct move_remove {
  configuration *config;
  triqs::mc_tools::random_generator &rng;
  double beta, U;

  dcomplex attempt() {
    auto k = config->perturbation_order();
    if (k <= 0) return 0;    // Config is empty, trying to remove makes no sense
    int p          = rng(k); // Choose one of the operators for removal
    auto det_ratio = config->Mmatrices[up].try_remove(p, p) * config->Mmatrices[down].try_remove(p, p);
    return -k / (beta * U) * det_ratio; // The Metropolis ratio
  }

  dcomplex accept() {
    for (auto &d : config->Mmatrices) d.complete_operation();
    return 1.0;
  }

  void reject() {
    for (auto &d : config->Mmatrices) d.reject_last_try(); // Finish insertion
  }                                                        // Nothing to do
};

//  -------------- QMC measurement ----------------

struct measure_M {

  configuration const *config;            // Pointer to the MC configuration
  block_gf<imtime, scalar_valued> &M_tau; // reference to M-matrix
  nda::array<dcomplex, 1> &M_hatree;      // Equal-time peak in M-matrix
  double beta;
  dcomplex Z = 0;
  long count = 0;

  measure_M(configuration const *config_, block_gf<imtime, scalar_valued> &M_tau_, nda::array<dcomplex, 1> &M_hatree_, double beta_)
     : config(config_), M_tau(M_tau_), M_hatree(M_hatree_), beta(beta_) {
    M_tau()    = 0;
    M_hatree() = 0;
  }

  void accumulate(dcomplex sign) {
    Z += sign;
    count++;

    // Loop over blocks
    for (auto s : {up, down}) {
      // Loop over every index pair (x,y) in the determinant matrix
      foreach (config->Mmatrices[s], [&](c_t const &c, cdag_t const &cdag, auto const &Ginv) {
        // Check for the equal-time case
        if (c.tau == cdag.tau) {
          M_hatree[s] += Ginv * sign;
        } else {
          // Project onto M_tau grid
          auto tau = c.tau - cdag.tau;
          if (tau >= 0)
            M_tau[s][closest_mesh_pt(tau)] += Ginv * sign;
          else // tau < 0, Account for anti-periodicity
            M_tau[s][closest_mesh_pt(tau + beta)] += -Ginv * sign;
        }
      })
        ;
    }
  }

  void collect_results(mpi::communicator const &c) {
    Z = mpi::all_reduce(Z, c);

    M_tau = mpi::all_reduce(M_tau, c);
    M_tau = M_tau / (-Z * beta);

    M_hatree = mpi::all_reduce(M_hatree, c);
    M_hatree = M_hatree / (-Z * beta);
    ;

    // Correct normalization for first and last bin
    for (auto s : {up, down}) {
      M_tau[s][0] *= 2.0;
      M_tau[s][M_tau[0].mesh().size() - 1] *= 2.0;
    }

    // Print the sign
    if (c.rank() == 0) std::cerr << "Average sign " << Z / c.size() / count << std::endl;
  }
};

// ------------ The main class of the solver ------------------------

ctint_solver::ctint_solver(double beta_, int n_iw, int n_tau) : beta(beta_) {

  G0_iw       = make_block_gf({"up", "down"}, gf<imfreq, scalar_valued>{{beta, Fermion, n_iw}, {}});
  G0tilde_tau = make_block_gf({"up", "down"}, gf<imtime, scalar_valued>{{beta, Fermion, n_tau}, {}});
  G0tilde_iw  = G0_iw;
  G_iw        = G0_iw;
  M_iw        = G0_iw;
  M_tau       = G0tilde_tau;
}

// The method that runs the qmc
void ctint_solver::solve(double U, double delta, int n_cycles, int length_cycle, int n_warmup_cycles, std::string random_name, int max_time) {

  mpi::communicator world;

  // Apply shift to G0_iw and Fourier transform
  nda::clef::placeholder<1> iw_;
  for (auto spin : {up, down}) {
    G0tilde_iw[spin](iw_) << 1.0 / (1.0 / G0_iw[spin](iw_) - U / 2);
    array<dcomplex, 1> mom{0, 1}; // Fix the moments: 0 + 1/omega
    G0tilde_tau()[spin] = triqs::gfs::fourier(G0tilde_iw[spin], make_const_view(mom));
  }

  // Rank-specific variables
  int verbosity   = (world.rank() == 0 ? 3 : 0);
  int random_seed = 34788 + 928374 * world.rank();

  // Construct a Monte Carlo loop
  triqs::mc_tools::mc_generic<dcomplex> CTQMC(random_name, random_seed, 1.0, verbosity);

  // Prepare the configuration
  auto config = configuration{G0tilde_tau, beta, delta};

  // Register moves and measurements
  CTQMC.add_move(move_insert{&config, CTQMC.get_rng(), beta, U}, "insertion");
  CTQMC.add_move(move_remove{&config, CTQMC.get_rng(), beta, U}, "removal");
  CTQMC.add_measure(measure_M{&config, M_tau, M_hatree, beta}, "M measurement");

  // Run and collect results
  CTQMC.warmup_and_accumulate(n_warmup_cycles, n_cycles, length_cycle, triqs::utility::clock_callback(max_time));
  CTQMC.collect_results(world);

  // Calculate M_iw from M_tau and M_hartree
  M_iw = make_gf_from_fourier(block_gf<imtime, scalar_valued>{M_tau}, G0_iw[0].mesh(), make_zero_tail(G0_iw));
  M_iw = make_hermitian(M_iw);
  for (auto s: {up, down}) M_iw[s](iw_) << M_iw[s](iw_) + M_hatree[s];

  // Compute the Green function from M_iw
  G_iw = G0tilde_iw + G0tilde_iw * M_iw * G0tilde_iw;
}
