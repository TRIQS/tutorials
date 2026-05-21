"""
04_gw_edmft.py
==============

Script for running a minimal GW+EDMFT calculation with CoQuí and TRIQS.

This script serves as the backbone for the tutorial `04_gw_edmft_minimum.ipynb`, 
which breaks it into stages and discusses each stage one by one. 

Workflow:
---------
1. Initialization: set imaginary-axis grids and build ModEST embedding objects.  
2. Downfolding (CoQuí): project lattice G, W onto the MLWF subspace.  
3. Impurity action: construct fermionic and bosonic Weiss fields.  
4. Impurity solution (TRIQS ModEST + CT-SEG): solve for Σ_imp, Π_imp.  

Notes:
------
- This script is a **minimum working example**, not a production run.  
- Input files (GW data, MLWFs, Coulomb integrals) must be prepared beforehand.  
- Parameters are chosen to run quickly for tutorial purposes.  
"""

# --- Imports -----------------------------------------------------------------
import sys
import numpy as np
from mpi4py import MPI

from h5 import HDFArchive
import triqs_modest as modest
from triqs.gfs import MeshImFreq

import coqui
from coqui.utils.imag_axes_ft import IAFT
import coqui.embed.edmft.utils as edmft_helper

# Local utilities in TRIQS school materials
sys.path.append("./utils/")
import ctseg_solver

# --- Initialization ---
# Path for pre-generated inputs 
mf_dir    = "data/qe_inputs/nio/555/out/"                 # QE outputs directory 
wan_h5    = "data/qe_inputs/nio/555/mlwf_dp/nio.mlwf.h5"  # MLWF file (8 orbitals for NiO)
coqui_dir = "data/coqui/nio/555/"                         # CoQuí checkpoint directory
coqui_h5  = "nio"                                         # CoQuí checkpoint with GW solution
thc_h5    = "data/coqui/nio/555/thc.coulomb.h5"           # THC Coulomb (optional: created if missing)

coqui_mpi = coqui.MpiHandler()
coqui.set_verbosity(coqui_mpi, output_level=1)

# Imaginary Mesh Setup
ir_kernel = IAFT(beta=100, wmax=5.0, prec="medium") 

# Embedding class from ModEST
print("-"*20+"One-Particle Embedding"+"-"*20)
E1 = modest.embedding_builder(
    spin_names=["up","down"], 
    block_decomposition=[[1,1], [1,1,1], [1,1,1]], 
    atom_to_imp=[0,1,2]
).drop(1).drop(1)
print(E1.description(True))

E2 = modest.embedding_builder(
    spin_names=["up","down"], 
    block_decomposition=[[2], [3], [3]],
    atom_to_imp=[0,1,2]
).drop(1).drop(1)
print(E2.description(True))

# Mean-field handler
mf_params = {
    "prefix": "nio", 
    "outdir": mf_dir, 
    "nbnd": 40
}
mf = coqui.make_mf(coqui_mpi, params=mf_params, mf_type="qe") 

# Coulomb Hamiltonian
thc_params = {
    "thresh": 1e-4, 
    "ecut": 60, 
    "save": thc_h5
}
thc = coqui.make_thc_coulomb(mf=mf, params=thc_params)

# --- Stage 1: Downfolding to the MLWF subspace  
# downfold for bare and dynamic screened interactions: V and W_t
wloc_params = {
    "outdir": coqui_dir,
    "prefix": coqui_h5, 
    "wannier_file": wan_h5,
    "input_type": "scf", 
    "screen_type": "gw_edmft_density", 
    "div_treatment": "gygi_extrplt"
}
# Vloc = (N_mlwf, N_mlwf, N_mlwf, N_mlwf)
# Wloc_t = (nts, N_mlwf, N_mlwf, N_mlwf, N_mlwf)
Vloc, Wloc_t = coqui.downfold_local_coulomb(
    h_int=thc, params=wloc_params, local_polarizabilities=None
) 

# downfold for the singple-particle Green's function
gloc_params = {
    "outdir": coqui_dir,
    "prefix": coqui_h5,
    "wannier_file": wan_h5,
    "input_type": "scf", 
}
Gloc_t = coqui.downfold_local_gf(mf=mf, params=gloc_params)
Gloc_t = np.repeat(Gloc_t, repeats=2, axis=1)


# --- Stage 2: Build the impurity problem ---
# --- Step 2.1: Extract impurity subspaces from MLWFs ---
gf_struct = E1.imp_block_shape[0]      # block structure of 1-e quantities
Gimp_t    = E1.extract_1p(Gloc_t)[0]   # block matrix
Vimp      = E2.extract_tensor(Vloc)[0] # (norb, norb, norb, norb) 
Wimp_t    = E2.extract_2p(Wloc_t)[0]   # (nts, norb, norb, norb, norb)

# --- Step 2.2: Compute the Weiss fields ---
# IR basis kernel (parameters must match CoQuí log)
ir_kernel = IAFT(beta=100, wmax=10.0, prec=1e-10) 

# Fermionic and bosonic Weiss fiels 
g_weiss_iw, u_weiss_iw = edmft_helper.compute_weiss_fields_w(
    ir_kernel = ir_kernel, 
    local_gf = {
        "Gloc_t": edmft_helper.blk_arr_to_arr(Gimp_t, gf_struct), 
        "Wloc_t": Wimp_t, 
        "Vloc": Vimp
    },
    impurity_selfenergies=None,
    density_only=True
)

# --- Step 2.3: Extract hybridization and h_0 (optional)
# h0: (nspin, norb, norb)
# delta_iw: (niw, nspin, norb, norb)
h0, delta_iw = edmft_helper.extract_h0_and_delta(g_weiss_iw, ir_kernel)



# --- Stage 3: Impurity Solver ---
# --- Step 3.1: Prepare TRIQS inputs from CoQuí outputs ---

# Set maximum fermionic Matsubara index from IR kernel
n_iw  = edmft_helper.set_n_iw(ir_kernel)     
n_tau = n_iw * 6 + 1                         # companion τ-grid length

# Fermionic and bosonic meshes consistent with IR kernel
iw_mesh_f = MeshImFreq(ir_kernel.beta, S='Fermion', n_iw=n_iw)
iw_mesh_b = MeshImFreq(ir_kernel.beta, S='Boson',   n_iw=n_iw)

# Convert CoQuí outputs to TRIQS containers
h0, delta_iw, h_int, u_weiss_iw = edmft_helper.to_triqs_containers(
    h0, delta_iw, Vimp, u_weiss_iw, ir_kernel,
    gf_struct = gf_struct,
    triqs_iw_mesh = {"fermion": iw_mesh_f, "boson": iw_mesh_b},
    density_hamiltonian = True,
    real_hamiltonian    = True
)

# --- Step 3.2: Solve the impurity and post-processing
solver_params = {
    "n_iw": n_iw,
    "n_tau": n_tau,
    "n_tau_bosonic": n_tau,
    "length_cycle": 300,
    "n_warmup_cycles": int(5e4),
    "n_cycles": int(2e5/coqui_mpi.comm_size()),
    "measure_pert_order": True,
    #"perform_tail_fit": True,
    #"fit_max_moment": 12,
    #"fit_min_w": 1.8,
    #"fit_max_w": 4.5
}
Res = ctseg_solver.solve_dynamic_imp(delta_iw, h0, u_weiss_iw, h_int, **solver_params)
