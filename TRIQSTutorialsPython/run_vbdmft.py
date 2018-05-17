from pytriqs.gf import *
from pytriqs.operators import *
from triqs_cthyb import Solver
from pytriqs.archive import HDFArchive
import pytriqs.utility.mpi as mpi
import numpy as np

import os
if mpi.is_master_node():
    if not os.path.exists('results_vbdmft'):
        os.makedirs('results_vbdmft')

# Parameters
t = 0.25
tp = -0.3*t
U = 10.*t
beta = 200./(4*t)

n_bins = 50
n_loops = 10

# Correspondence between doping and mu
mu_table = {}
mu_table['0.04'] = 0.615544
mu_table['0.08'] = 0.504525
mu_table['0.12'] = 0.412874
mu_table['0.16'] = 0.339201
mu_table['0.20'] = 0.271545
mu_table['0.32'] = 0.080221
mu_table['0.48'] = -0.131217

# Get the dispersion over the BZ
k_linear = np.linspace(-np.pi, np.pi, 1000)
kx, ky = np.meshgrid(k_linear, k_linear)
epsk = -2 * t * (np.cos(kx) + np.cos(ky)) - 4 * tp * np.cos(kx) * np.cos(ky)

# A mask giving the k points inside the central patch
in_central_patch = (np.abs(kx) < np.pi/np.sqrt(2)) & (np.abs(ky) < np.pi/np.sqrt(2))

# Find the partial densities of states associated to the patches
energies, epsilon, rho, delta = {},{}, {}, {}
energies['even'] = np.extract(in_central_patch, epsk)
energies['odd'] = np.extract(np.invert(in_central_patch), epsk)
for patch in ['even','odd']:
    h = np.histogram(energies[patch], bins=n_bins, normed=True)
    epsilon[patch] = 0.5 * (h[1][0:-1] + h[1][1:])
    rho[patch] = h[0]
    delta[patch] = h[1][1]-h[1][0]

# Construct the impurity solver
S = Solver(beta = beta,
           gf_struct = [('even-up',[0]), ('odd-up',[0]), ('even-down',[0]), ('odd-down',[0])], n_l = 100)

# The local lattice Green's function
G = S.G0_iw.copy()

# Rotation
cn, cn_dag, nn = {}, {}, {}
for spin in ['up','down']:
    cn['1-%s'%spin] = (c('even-%s'%spin,0) + c('odd-%s'%spin,0)) / np.sqrt(2)
    cn['2-%s'%spin] = (c('even-%s'%spin,0) - c('odd-%s'%spin,0)) / np.sqrt(2)
    nn['1-%s'%spin] = dagger(cn['1-%s'%spin]) * cn['1-%s'%spin]
    nn['2-%s'%spin] = dagger(cn['2-%s'%spin]) * cn['2-%s'%spin]

# Local Hamiltonian
h_loc = U * (nn['1-up'] * nn['1-down'] + nn['2-up'] * nn['2-down'])

# Run for several dopings
for doping, mu in mu_table.items():

    # Initial guess
    S.Sigma_iw << mu

    # DMFT loop
    for it in range(n_loops):
    
        # DCA self-consistency - get local lattice G
        G.zero()
        for spin in ['up', 'down']:
            for patch in ['even', 'odd']:
                # Hilbert transform
                for i in range(n_bins):
                    G['%s-%s'%(patch,spin)] += rho[patch][i] * delta[patch] * inverse(iOmega_n + mu - epsilon[patch][i] - S.Sigma_iw['%s-%s'%(patch,spin)])

        # DCA self-consistency - find next impurity G0
        for block, g0 in S.G0_iw:
            g0 << inverse(inverse(G[block]) + S.Sigma_iw[block])

        # Run the solver. The results will be in S.G_tau, S.G_iw and S.G_l
        S.solve(h_int = h_loc,                           # Local Hamiltonian
                n_cycles  = 200000,                      # Number of QMC cycles
                length_cycle = 50,                       # Length of one cycle
                n_warmup_cycles = 10000,                 # Warmup cycles
                measure_g_l = True)                      # Measure G_l
    
        if mpi.is_master_node():
            with HDFArchive("results_vbdmft/doping_%s.h5"%doping) as A:
                A['G_iw-%i'%it] = S.G_iw
                A['Sigma_iw-%i'%it] = S.Sigma_iw
                A['G0_iw-%i'%it] = S.G0_iw
