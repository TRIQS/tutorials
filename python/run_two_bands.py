from pytriqs.gf.local import *
from pytriqs.operators import *
from pytriqs.archive import *
from pytriqs.applications.impurity_solvers.cthyb import Solver
from itertools import product
import numpy as np

import os
if not os.path.exists('results_two_bands'):
    os.makedirs('results_two_bands')

# Parameters of the model
t = 1.0
beta = 10.0
n_loops = 10
filling = 'quarter' #'half'
n_orbitals = 2

# Construct the segment solver
S = Solver(beta = beta, gf_struct = {'up-0':[0], 'up-1':[0], 'down-0':[0], 'down-1':[0]})

for coeff in [0.0, 0.1, 0.2]:
#for coeff in [0.1]:

    # Run for several values of U
    for U in np.arange(1.0, 13.0, 1.0):
    #for U in [1.0]:

        J = coeff * U

        # Expression of mu for half and quarter filling
        if filling == 'half':
            mu = 0.5*U + 0.5*(U-2*J) + 0.5*(U-3*J)
        elif filling == 'quarter':
            mu = -0.81 + (0.6899-1.1099*coeff)*U + (-0.02548+0.02709*coeff-0.1606*coeff**2)*U**2


        # Set the interacting Kanamori hamiltonian
        h_int = Operator()
        for o in range(0,n_orbitals):
            h_int += U*n('up-%s'%o,0)*n('down-%s'%o,0)
        
        for o1,o2 in product(range(0,n_orbitals),range(0,n_orbitals)):
            if o1==o2: continue
            h_int += (U-2*J)*n('up-%s'%o1,0)*n('down-%s'%o2,0)
        
        for o1,o2 in product(range(0,n_orbitals),range(0,n_orbitals)):
            if o2>=o1: continue;
            h_int += (U-3*J)*n('up-%s'%o1,0)*n('up-%s'%o2,0)
            h_int += (U-3*J)*n('down-%s'%o1,0)*n('down-%s'%o2,0)
        
        for o1,o2 in product(range(0,n_orbitals),range(0,n_orbitals)):
            if o1==o2: continue
            h_int += -J*c_dag('up-%s'%o1,0)*c_dag('down-%s'%o1,0)*c('up-%s'%o2,0)*c('down-%s'%o2,0)
            h_int += -J*c_dag('up-%s'%o1,0)*c_dag('down-%s'%o2,0)*c('up-%s'%o2,0)*c('down-%s'%o1,0)


        # This is a first guess for G
        S.G0_iw << inverse(iOmega_n + mu - t**2 * SemiCircular(2*t))

        # DMFT loop with self-consistency
        for i in range(n_loops):

            print "\n\nIteration = %i / %i" % (i+1, n_loops)

            # Symmetrize the Green's function and use self-consistency
            if i > 0:
                g = 0.25 * ( S.G_iw['up-0'] + S.G_iw['up-1'] + S.G_iw['down-0'] + S.G_iw['down-1'] )
                for name, g0 in S.G0_iw:
                    g0 <<= inverse(iOmega_n + mu - t**2 * g)

            # Solve the impurity problem
            S.solve(h_int = h_int,
                    n_cycles  = 30000,
                    length_cycle = 100,
                    n_warmup_cycles = 5000)

            # Check density
            for name, g in S.G_iw:
                print name, ": ",g.density()[0,0].real

            # Save iteration in archive
            with HDFArchive("results_two_bands/%s-U%.2f-J%.2f.h5"%(filling,U,J)) as A:
                A['G-%i'%i] = S.G_iw
                A['Sigma-%i'%i] = S.Sigma_iw
