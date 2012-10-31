import numpy

from pytriqs.Base.GF_Local import *
from pytriqs.Solvers import Solver_Base

class Solver(Solver_Base):

    """A simple IPT solver for the symmetric one band Anderson model"""
    def __init__(self, **params):
        self.Name = 'Iterated Perturbation Theory'

        self.U = params['U']
        self.beta = params['beta']

        # Only one block in GFs
        g = GFBloc_ImFreq(Indices=[0], Beta=self.beta, Name='0')
        self.G = GF(NameList=('0',), BlockList=(g,))
        self.G0 = self.G.copy()

    def Solve(self):

        # Imaginary time representation of G_0
        g0t = GFBloc_ImTime(Indices=[0], Beta=self.beta, Name='0')
        G0t = GF(NameList=('0',), BlockList=(g0t,))
        G0t['0'].setFromInverseFourierOf(self.G0['0'])

        # IPT expression for the self-energy (particle-holy symmetric case is implied)
        Sigmat = G0t.copy()
        Sigmat['0'] <<= (self.U**2)*G0t['0']*G0t['0']*G0t['0']
        
        self.Sigma = self.G0.copy()
        self.Sigma['0'].setFromFourierOf(Sigmat['0'])

        # New impurity GF from the Dyson's equation
        self.G <<= self.G0*inverse(1.0 - self.Sigma*self.G0)

S = 0

def run(**params):
    """IPT loop"""

    # Read input parameters
    U = params['U']
    beta = params['beta']
    N_loops = params['N_loops']
    Initial_G0 = params['Initial_G0']
    Self_Consistency = params['Self_Consistency']

    global S
    # Create a new IPT solver object
    S = Solver(U=U, beta=beta)
    # Initialize the bare GF using the function passed in through Initial_G0 parameter
    Initial_G0(S.G0)

    # DMFT iterations
    for IterationNumber in range(N_loops):
        S.Solve()
        Self_Consistency(S.G0,S.G)
