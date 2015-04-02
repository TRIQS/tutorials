from pytriqs.gf.local import *
from pytriqs.operators import *
from pytriqs.archive import *
from pytriqs.applications.impurity_solvers.cthyb import Solver
from numpy import *

# Parameters of the model
t = 1.0
beta = 10.0
n_loops = 10

# Construct the impurity solver
S = Solver(beta = beta, gf_struct = {'up':[0], 'down':[0]})

# I run for several values of U
for U in arange(1.0, 13.0):

    # This is a first guess for G
    S.G_iw << SemiCircular(2*t)

    # DMFT loop with self-consistency
    for i in range(n_loops):
    
        print "\n\nIteration = %i / %i" % (i+1, n_loops)
    
        # Symmetrize the Green's function and use self-consistency
        g = 0.5 * ( S.G_iw['up'] + S.G_iw['down'] )
        for name, g0 in S.G0_iw:
            g0 << inverse( iOmega_n + U/2.0 - t**2 * g )

        # Solve the impurity problem
        S.solve(h_loc = U * n('up',0) * n('down',0),   # Local Hamiltonian 
            n_cycles  = 10000,                           # Number of QMC cycles
            n_warmup_cycles = 5000,                      # Warmup cycles
            )
    
        # Save iteration in archive
        A = HDFArchive("single-U%.2f.h5"%U)
        A['G-%i'%i] = S.G_iw
        A['Sigma-%i'%i] = S.Sigma_iw
        del A
