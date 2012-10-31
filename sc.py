# Visualization of the Mott transition

from math import *
import os
import numpy

from pytriqs.Base.GF_Local import *
from pytriqs.Base.GF_Local import Omega, SemiCircular, inverse
from pytriqs.Base.Archive import *
from pytriqs.Base.Plot.MatplotlibInterface import oplot
import matplotlib.pyplot as plt

import IPT

beta = 40      # Inverse temperature
U = numpy.arange(0,4.05,0.1)    # Range of interaction constants to scan
t = 0.5     # Scaled hopping constant on the Bethe lattice

N_loops = 20    # Number of DMFT loops

# Pade-related
DOSMesh = numpy.arange(-4,4,0.02)   # Mesh to plot densities of states
eta = 0.00     # Imaginary frequency offset to use with Pade
Pade_L = 201    # Number of Matsubara frequencies to use with Pade

# Bare Green's function of the Bethe lattice (semicircle) 
def Initial_G0(G0):
    G0 <<= SemiCircular(2*t)

# Self-consistency condition
def Self_Consistency(G0,G):
    G0['0'] <<= inverse(Omega - (t**2)*G['0'])

# Save results to an HDF5-archive
ar = HDF_Archive('Mott.h5','w')
ar['beta'] = beta
ar['t'] = t
    
# List of files with DOS figures
DOS_files=[]
    
# Scan over values of U
for u in U:
    print "Running DMFT calculation for beta=%.2f, U=%.2f, t=%.2f" % (beta,u,t)
    IPT.run(N_loops = N_loops, beta=beta, U=u, Initial_G0=Initial_G0, Self_Consistency=Self_Consistency)
    
    # The resulting local GF on the real axis
    g_real = GFBloc_ReFreq(Indices = [0], Beta = beta, MeshArray = DOSMesh, Name = '0')
    G_real = GF(NameList = ('0',), BlockList = (g_real,), Copy = True)
    
    # Analytic continuation with Pade
    G_real['0'].setFromPadeOf(IPT.S.G['0'], N_Matsubara_Frequencies=Pade_L, Freq_Offset=eta)

    # Save data to the archive
    ar['U' + str(u)] = {'G0': IPT.S.G0, 'G': IPT.S.G, 'Sigma': IPT.S.Sigma, 'G_real':G_real}
   
    # Plot the DOS
    fig = plt.figure()
    oplot(G_real['0'][0,0], RI='S', Name="DOS", figure = fig)
    
    # Adjust 'y' axis limits accordingly to the Luttinger sum rule 
    fig.axes[0].set_ylim(0,1/pi/t*1.1)
    
    # Set title of the plot
    fig_title = "Local DOS, IPT, Bethe lattice, $\\beta=%.2f$, $U=%.2f$" % (beta,u)
    plt.title(fig_title)

    # Save the figure as a PNG file
    DOS_file = "DOS_beta%.2fU%.2f.png" % (beta,u)
    fig.savefig(DOS_file, format="png", transparent=False)
    DOS_files.append(DOS_file)
    plt.close(fig)

# Create an animated GIF
# (you need to have 'convert' utility installed; it is a part of ImageMagick suite) 
convert_cmd = "convert -delay 25 -loop 1"
convert_cmd += " " + ' '.join(DOS_files)
convert_cmd += " " + "DOS.gif"

print "Creating an animated DOS plot..."
os.system(convert_cmd)
