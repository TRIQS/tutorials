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

  configuration(block_gf<imtime, scalar_valued> &g0tilde_tau, double beta, double delta) {
    // Initialize the M-matrices. 100 is the initial matrix size
    for (auto spin : {up, down}) Mmatrices.emplace_back(g0bar_tau{g0tilde_tau[spin], beta, delta, spin}, 100);
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

  configuration const *config;         // Pointer to the MC configuration
  block_gf<imfreq, scalar_valued> &Mw; // reference to M-matrix
  double beta;
  dcomplex Z = 0;
  long count = 0;

  measure_M(configuration const *config_, block_gf<imfreq, scalar_valued> &Mw_, double beta_) : config(config_), Mw(Mw_), beta(beta_) { Mw() = 0; }

  void accumulate(dcomplex sign) {
    Z += sign;
    count++;

    for (auto spin : {up, down}) {

      // A lambda to measure the M-matrix in frequency
      auto lambda = [this, spin, sign](c_t const &c, cdag_t const &cdag, dcomplex M) {
        auto const &mesh = this->Mw[spin].mesh();
        auto phase_step  = -1i * M_PI * (c.tau - cdag.tau) / beta;
        auto coeff       = std::exp((2 * mesh.first_index() + 1) * phase_step);
        auto fact        = std::exp(2 * phase_step);
        for (auto const &om : mesh) {
          this->Mw[spin][om] += sign * M * coeff;
          coeff *= fact;
        }
      };

      foreach (config->Mmatrices[spin], lambda)
        ;
    }
  }

  void collect_results(mpi::communicator const &c) {
    Mw = mpi::all_reduce(Mw, c);
    Z  = mpi::all_reduce(Z, c);
    Mw = Mw / (-Z * beta);

    // Print the sign
    if (c.rank() == 0) std::cerr << "Average sign " << Z / c.size() / count << std::endl;
  }
};

// ------------ The main class of the solver ------------------------

ctint_solver::ctint_solver(double beta_, int n_iw, int n_tau) : beta(beta_) {

  g0_iw       = make_block_gf({"up", "down"}, gf<imfreq, scalar_valued>{{beta, Fermion, n_iw}, {}});
  g0tilde_tau = make_block_gf({"up", "down"}, gf<imtime, scalar_valued>{{beta, Fermion, n_tau}, {}});
  g0tilde_iw  = g0_iw;
  g_iw        = g0_iw;
  M_iw        = g0_iw;
}

// The method that runs the qmc
void ctint_solver::solve(double U, double delta, int n_cycles, int length_cycle, int n_warmup_cycles, std::string random_name, int max_time) {

  mpi::communicator world;

  // Apply shift to g0_iw and Fourier transform
  nda::clef::placeholder<1> om_;
  for (auto spin : {up, down}) {
    g0tilde_iw[spin](om_) << 1.0 / (1.0 / g0_iw[spin](om_) - U / 2);
    array<dcomplex, 1> mom{0, 1}; // Fix the moments: 0 + 1/omega
    g0tilde_tau()[spin] = triqs::gfs::fourier(g0tilde_iw[spin], make_const_view(mom));
  }

  // Rank-specific variables
  int verbosity   = (world.rank() == 0 ? 3 : 0);
  int random_seed = 34788 + 928374 * world.rank();

  // Construct a Monte Carlo loop
  triqs::mc_tools::mc_generic<dcomplex> CTQMC(random_name, random_seed, 1.0, verbosity);

  // Prepare the configuration
  auto config = configuration{g0tilde_tau, beta, delta};

  // Register moves and measurements
  CTQMC.add_move(move_insert{&config, CTQMC.get_rng(), beta, U}, "insertion");
  CTQMC.add_move(move_remove{&config, CTQMC.get_rng(), beta, U}, "removal");
  CTQMC.add_measure(measure_M{&config, M_iw, beta}, "M measurement");

  // Run and collect results
  CTQMC.warmup_and_accumulate(n_warmup_cycles, n_cycles, length_cycle, triqs::utility::clock_callback(max_time));
  CTQMC.collect_results(world);

  // Compute the Green function from Mw
  g_iw = g0tilde_iw + g0tilde_iw * M_iw * g0tilde_iw;
}
