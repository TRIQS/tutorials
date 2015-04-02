from pytriqs.gf.local import *
from pytriqs.operators import *
from pytriqs.archive import *
from pytriqs.applications.impurity_solvers.cthyb_segment import Solver
from numpy import *

# Parameters of the model
t = 1.0
beta = 10.0
n_loops = 10
filling = 'half'

# Construct the segment solver
S = Solver(beta = beta, block_names = ['up-0', 'up-1', 'down-0', 'down-1'])

# We will need these later
F = S.G0.copy()
G = S.G0.copy()
Sigma = S.G0.copy()

for coeff in [0.0, 0.1, 0.2]:

    # I run for several values of U
    for U in arange(1.0, 13.0, 1.0):

        J = coeff * U

        # Expression of mu for half and quarter filling
        if filling == 'half':
            mu = 0.5*U + 0.5*(U-2*J) + 0.5*(U-3*J)
        elif filling == 'quarter':
            mu = -0.81 + (0.6899-1.1099*coeff)*U + (-0.02548+0.02709*coeff-0.1606*coeff**2)*U**2

        # Interaction matrix
        Umat = zeros([4,4])
        for i in range(2):
            for j in range(2):
                Umat[i][j] = U-3*J if i != j else 0.0
                Umat[i+2][j+2] = U-3*J if i != j else 0.0
                Umat[i][j+2] = U if i == j else U-2*J
                Umat[i+2][j] = U if i == j else U-2*J

        # This is a first guess for G
        S.G0 <<= inverse(iOmega_n + mu - t**2 * SemiCircular(2*t))

        # DMFT loop with self-consistency
        for i in range(n_loops):

            print "\n\nIteration = %i / %i" % (i+1, n_loops)

            # Symmetrize the Green's function and use self-consistency
            if i>0:
                g = 0.25 * ( G['up-0'] + G['down-0'] + G['up-1'] + G['down-1'] )
                for name, g0 in S.G0:
                    g0 <<= inverse(iOmega_n + mu - t**2 * g)

            # Solve the impurity problem
            S.solve(U = Umat,
                    n_cycles  = 30000,
                    length_cycle = 100,
                    n_warmup_cycles = 5000,
                    improved_estimator = True)

            # Symmetrize F
            f_tau = 0.25 * ( S.F_tau['up-0'] + S.F_tau['down-0'] + S.F_tau['up-1'] + S.F_tau['down-1'] )

            # Get G and Sigma
            for name, f in F:
                f <<= Fourier(f_tau)
                G[name] <<= S.G0[name] + S.G0[name] * f
            Sigma = inverse(S.G0) - inverse(G)

            # Check density
            print G.density()

            # Save iteration in archive
            A = HDFArchive("results-%s-U%.2f-J%.2f.h5"%(filling,U,J))
            A['G-%i'%i] = G
            A['Sigma-%i'%i] = Sigma
            del A
