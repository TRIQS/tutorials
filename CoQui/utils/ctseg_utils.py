import triqs.utility.mpi as mpi
import numpy as np
from itertools import product

from triqs.gfs import (iOmega_n, MeshImFreq, Gf, BlockGf, make_gf_imfreq, make_gf_imtime, 
                      make_gf_from_fourier, Idx, make_hermitian, is_gf_hermitian)
from triqs.gfs.gf_fnt import fit_hermitian_tail_on_window, replace_by_tail
from triqs.gfs.tools import inverse, make_zero_tail
from triqs.gfs.descriptors import Fourier
from triqs.operators.util.extractors import block_matrix_from_op
from triqs.operators.util.U_matrix import reduce_4index_to_2index

from triqs_ctseg import Solver as CTSEG_Solver

def post_process(solver, **post_proc_params):
    
    post_process_sigma(solver, **post_proc_params)
    
    if solver.D0_tau is not None and solver.results.nn_tau is not None:
        post_process_pi(solver)


def post_process_pi(solver):
    mpi.report("Charge susceptibility is measured for a impurity with dyanmic interations")
    mpi.report('--> Post-processing the density-density susceptibility to obtain the impurity polarizability.\n')

    n_color = 0
    for _, blk_dim in solver.gf_struct:
        n_color += blk_dim
    assert n_color % 2 == 0, "Oh oh... n_color is an odd number." 
    n_orb = n_color// 2 

    block_name       = []
    index_in_block   = []
    color_to_orbital = []  # mapping from color to orbital
    for color in range(n_color):
        block_name.append(find_block_name(color, solver.gf_struct))
        index_in_block.append(find_index_in_block(color, solver.gf_struct))
        color_to_orbital.append(find_orbital_index(color, solver.gf_struct))
    
    # initializaiton 
    iw_mesh = MeshImFreq(beta = solver.beta, statistic="Boson", n_iw = solver.n_iw)
    tau_mesh = solver.results.nn_tau[block_name[0], block_name[0]].mesh

    D0_tau = Gf(mesh=tau_mesh, target_shape=(n_color, n_color))
    nn_tau = D0_tau.copy()
    for c1 in range(n_color):
        for c2 in range(n_color):
            nn_tau.data[:, c1, c2] = (
                solver.results.nn_tau[block_name[c1], block_name[c2]].data[:, index_in_block[c1], index_in_block[c2]])
            D0_tau.data[:, c1, c2] = (
                solver.D0_tau[block_name[c1], block_name[c2]].data[:, index_in_block[c1], index_in_block[c2]])

    
    # density for the constant part of chi
    densities = np.zeros(n_color, dtype=float)
    for c1 in range(n_color):
        densities[c1] = solver.results.densities[block_name[c1]][index_in_block[c1]]
    mpi.report(f"Average of time-dependent occupations: {densities}")

    mpi.report("Subtracting the constant component, and then symmetrizing the density-density susceptibility: \n"
               "  1. nn(t).imag = 0.0\n"
               "  2. nn(i, j) = nn(j, i)\n")

    for c1, c2 in product(range(n_color), repeat=2):
        if c1 >= c2:
            # remove the constant part 
            nn_tau[c1,c2].data[:] -= (densities[c1] * densities[c2])
            # symmetrization
            nn_tau[c1,c2].data.imag = 0.0
            if c1 != c2:
                nn_tau[c2,c1].data[:] -= (densities[c2] * densities[c1])
                nn_tau[c1,c2].data[:] += nn_tau[c2,c1].data[:]
                nn_tau[c1,c2].data[:] /= 2.0
                nn_tau[c2,c1] << nn_tau[c1,c2]

    nn_iw = Gf(mesh = iw_mesh, target_shape = nn_tau.target_shape)
    U_iw = nn_iw.copy()
    nn_known_moments = make_zero_tail(nn_iw, n_moments=2)
    nn_iw.set_from_fourier(nn_tau, nn_known_moments)
    U_iw.set_from_fourier(D0_tau, nn_known_moments)
    #assert is_gf_hermitian(nn_iw)
    #nn_iw << make_hermitian(nn_iw)

    # convert to density-density basis 
    nn_iw_dd = Gf(mesh=nn_iw.mesh, target_shape=[n_orb, n_orb])
    U_iw_dd = nn_iw_dd.copy()
    for c1, c2 in product(range(n_color), repeat=2):
        nn_iw_dd[color_to_orbital[c1], color_to_orbital[c2]].data[:] += nn_iw[c1, c2].data[:]
        # multiply by 0.25 to take average over (up, up), (up, dn), (dn, up) and (dn, dn) 
        # (assuming all sub blocks are the same)
        U_iw_dd[color_to_orbital[c1], color_to_orbital[c2]].data[:] += U_iw[c1, c2].data[:] * 0.25
        
    # Convert to a product basis 
    nn_iw_pb = Gf(mesh=nn_iw.mesh, target_shape=[n_orb*n_orb, n_orb*n_orb])
    for i, j in product(range(n_orb), repeat=2):
        if i >= j:
            # only the real part
            nn_iw_pb[i*n_orb+i, j*n_orb+j] << nn_iw_dd[i,j].real
            if i != j:
                nn_iw_pb[j*n_orb+j, i*n_orb+i] << nn_iw_pb[i*n_orb+i, j*n_orb+j]


    # Vijkl is the full 4-index tensor without the block structure 
    U_iw_pb = Gf(mesh=nn_iw_pb.mesh, target_shape=nn_iw_pb.target_shape)
    # Vijkl is in TRIQS's notation
    Vijkl = extract_Uijkl_from_h_int(h_int=solver.h_int, gf_struct=solver.gf_struct)
    for i, j in product(range(n_orb), repeat=2):
        if i == j:
            # intra-orbital density-density term
            U_iw_pb[i*n_orb+i, i*n_orb+i] << U_iw_dd[i, i].real
            U_iw_pb[i*n_orb+i, i*n_orb+i].data[:] += Vijkl[i, i, i, i]
        if i > j:
            # inter-orbital density-density term
            U_iw_pb[i*n_orb+i, j*n_orb+j] << U_iw_dd[i, j]
            U_iw_pb[i*n_orb+i, j*n_orb+j].data[:] += Vijkl[i, j, i, j]
            U_iw_pb[j*n_orb+j, i*n_orb+i] << U_iw_pb[i*n_orb+i, j*n_orb+j]
            # Hund's J: Spin-flip (i, j, j, i)
            U_iw_pb[j*n_orb+i, j*n_orb+i].data[:] += Vijkl[i, j, j, i]
            U_iw_pb[i*n_orb+j, i*n_orb+j] << U_iw_pb[j*n_orb+i, j*n_orb+i]
            # Hund's J: Pair hopping (i, j, i, j)
            U_iw_pb[j*n_orb+i, i*n_orb+j].data[:] += Vijkl[i, i, j, j]
            U_iw_pb[i*n_orb+j, j*n_orb+i] << U_iw_pb[j*n_orb+i, i*n_orb+j]
    

    # Dyson equation: Pi(w) = [U(w)*Chi(w) - I]^-1 * Chi(w)
    Pi_iw_pb = Gf(mesh=nn_iw_pb.mesh, target_shape=nn_iw_pb.target_shape)
    ones = np.eye(n_orb*n_orb, dtype=complex)
    for iwn in nn_iw_pb.mesh:
        denom = U_iw_pb[iwn] @ nn_iw_pb[iwn] - ones
        cond = np.linalg.cond(denom)
        if cond > 20: 
            mpi.report(f"WARNING: Large condition number for [U(w) * Chi(w) - I] = {cond} at n = {iwn.index}.")        
        Pi_iw_pb[iwn] = np.linalg.pinv(denom) @ nn_iw_pb[iwn]
        # explicit set Pi(iw).imag = 0.0
        Pi_iw_pb[iwn].imag = 0.0


    # Screened interaction W(w) = U(w) - U(w) * Chi(w) * U(w)
    W_iw_pb = Gf(mesh=nn_iw_pb.mesh, target_shape=nn_iw_pb.target_shape)
    for iwn in nn_iw_pb.mesh:
        W_iw_pb[iwn] = U_iw_pb[iwn] - U_iw_pb[iwn] @ nn_iw_pb[iwn] @ U_iw_pb[iwn]
    
    # Remove the static part 
    for i, j in product(range(n_orb), repeat=2):
        if i == j:
            # intra-orbital density-density term
            W_iw_pb[i*n_orb+i, i*n_orb+i].data[:] -= Vijkl[i, i, i, i]
        if i > j:
            # inter-orbital density-density term
            W_iw_pb[i*n_orb+i, j*n_orb+j].data[:] -= Vijkl[i, j, i, j]
            W_iw_pb[j*n_orb+j, i*n_orb+i] << W_iw_pb[i*n_orb+i, j*n_orb+j]
            # Hund's J: Spin-flip (i, j, j, i)
            W_iw_pb[j*n_orb+i, j*n_orb+i].data[:] -= Vijkl[i, j, j, i]
            W_iw_pb[i*n_orb+j, i*n_orb+j] << W_iw_pb[j*n_orb+i, j*n_orb+i]
            # Hund's J: Pair hopping (i, j, i, j)
            W_iw_pb[j*n_orb+i, i*n_orb+j].data[:] -= Vijkl[i, i, j, j]
            W_iw_pb[i*n_orb+j, j*n_orb+i] << W_iw_pb[j*n_orb+i, i*n_orb+j]

    # transform back to 4-index tensor 
    solver.Pi_iw = Gf(mesh=nn_iw_pb.mesh, target_shape=(n_orb, n_orb, n_orb, n_orb))
    solver.W_iw = solver.Pi_iw.copy()
    for i, j, k, l in product(range(n_orb), repeat=4):
        solver.Pi_iw[i,j,k,l] << Pi_iw_pb[i*n_orb+j, k*n_orb+l]
        solver.W_iw[i,j,k,l] << W_iw_pb[i*n_orb+j, k*n_orb+l]


def post_process_sigma(solver, **post_proc_params):
    
    # initializaiton 
    mesh = MeshImFreq(beta = solver.beta, statistic="Fermion", n_iw = solver.n_iw)
    solver.Sigma_iw = BlockGf(mesh = mesh, gf_struct = solver.gf_struct)
    solver.Sigma_iw.zero()
    solver.Sigma_moments = None
    solver.G_iw = solver.Sigma_iw.copy()
    solver.G0_iw = solver.Sigma_iw.copy()
    
    # Fourier transform G(tau) to G(iw)
    Gf_known_moments = make_zero_tail(solver.G_iw, n_moments=2)
    for i, bl in enumerate(solver.G_iw.indices):
        Gf_known_moments[i][1] = np.eye(solver.G_iw[bl].target_shape[0])
        solver.G_iw[bl].set_from_fourier(solver.results.G_tau[bl], Gf_known_moments[i])
    assert is_gf_hermitian(solver.G_iw)
    solver.G_iw << make_hermitian(solver.G_iw)

    # compute fermionic Weiss field g(iw)
    Delta_iw = BlockGf(mesh = mesh, gf_struct = solver.gf_struct)
    Delta_known_moments = make_zero_tail(Delta_iw, n_moments=1)
    for i, bl in enumerate(solver.Delta_tau.indices):
        Delta_iw[bl].set_from_fourier(solver.Delta_tau[bl], Delta_known_moments[i])
        solver.G0_iw[bl] << inverse(iOmega_n - Delta_iw[bl] - solver.h_loc0_mat[i])
    #assert is_gf_hermitian(solver.G0_iw)
    solver.G0_iw << make_hermitian(solver.G0_iw)

    # Compute the HF self-energy as the first moment of the self-energy
    solver.Sigma_Hartree = {}
    solver.Sigma_moments = {}
    mpi.report('\nEvaluating static impurity self-energy analytically using CT-SEG interacting density:')
    # Uijkl is the full 4-index tensor without the block structure 
    Vijkl = extract_Uijkl_from_h_int(h_int=solver.h_int, gf_struct=solver.gf_struct)
    Uw0_ijkl = Vijkl + extract_screen_matrix_from_D0_tau(blk2_D0_tau=solver.D0_tau, gf_struct=solver.gf_struct)
    norb = Vijkl.shape[0]

    # compute density matrix without the block structure
    density_matrix = {"up": np.zeros((norb, norb)), 
                      "down": np.zeros((norb,norb))}
    o1_up, o1_dn = 0, 0
    for blk_name, blk_dim in solver.gf_struct:
        spin = "up" if blk_name[:2] == "up" else "down"
        o1 = o1_up if blk_name[:2] == "up" else o1_dn
        for iorb in range(blk_dim):
            density_matrix[spin][o1+iorb, o1+iorb] = solver.results.densities[blk_name][iorb]
        o1 += blk_dim
        assert o1 <= norb, "orbital offset excceds band range"
        if blk_name[:2] == "up":
            o1_up = o1 % norb
        else:
            o1_dn = o1 % norb

    # compute HF self-energy
    o1_up, o1_dn = 0, 0
    for blk_name, blk_dim in solver.gf_struct:
        solver.Sigma_Hartree[blk_name] = np.zeros((blk_dim, blk_dim), dtype=float)

        spin = "up" if blk_name[:2] == "up" else "down"
        o1 = o1_up if blk_name[:2] == "up" else o1_dn

        # Sigma_HF_{ij} = \sum_{a,b} n_{ab} \left( 2 Uw0_{i a j b} - U_{i a b j} \right)
        for iorb, jorb in product(range(blk_dim), repeat=2):
            # inner needs to run over the entire norb
            for inner in range(norb):
                # exchange diagram K
                solver.Sigma_Hartree[blk_name][iorb, jorb] -= (
                    density_matrix[spin][inner, inner].real * (Vijkl[o1+iorb, inner, inner, o1+jorb].real))                 
                # Hartree (Coulomb) diagram J
                solver.Sigma_Hartree[blk_name][iorb, jorb] += (
                    density_matrix[spin][inner, inner].real * (2 * Uw0_ijkl[o1+iorb, inner, o1+jorb, inner].real))
        
        o1 = o1 + blk_dim
        assert o1 <= norb, "orbital offset excceds band range"
        if blk_name[:2] == "up":
            o1_up = o1 % norb
        else:
            o1_dn = o1 % norb

    # create moments array from this
    for blk_name, hf_val in solver.Sigma_Hartree.items():
        mpi.report(f"Σ_HF {blk_name}:")
        mpi.report(f"    {hf_val}")
        solver.Sigma_moments[blk_name] = np.array([hf_val], dtype=complex)
    mpi.report("")
    
    # Compute Self-energy
    if solver.results.F_tau is None:
        mpi.report("F(tau) is not measured -> Compute the self-energy via the Dyson equation.\n")
        solver.Sigma_iw = inverse(solver.G0_iw) - inverse(solver.G_iw)
    else:
        mpi.report("F(tau) is measured -> Compute the self-energy via the improved estimator.\n")
        F_iw = solver.G_iw.copy()
        F_iw << 0.0
        F_known_moments = make_zero_tail(F_iw, n_moments=1)
        for i, bl in enumerate(F_iw.indices):
            F_iw[bl].set_from_fourier(solver.results.F_tau[bl], F_known_moments[i])

        for block, fw in F_iw:
            for iw in fw.mesh:
                solver.Sigma_iw[block][iw] = fw[iw] / solver.G_iw[block][iw]

    if post_proc_params['perform_tail_fit']:
        # tail fitting for the self-energy 
        solver.Sigma_iw = tail_fit(
            solver.Sigma_iw,
            fit_min_n=post_proc_params['fit_min_n'],
            fit_max_n=post_proc_params['fit_max_n'],
            fit_min_w=post_proc_params['fit_min_w'],
            fit_max_w=post_proc_params['fit_max_w'],
            fit_max_moment=post_proc_params['fit_max_moment'],
            fit_known_moments=solver.Sigma_moments
        )

    solver.Sigma_dynamic = solver.Sigma_iw.copy()
    for bl, g in solver.Sigma_dynamic: solver.Sigma_dynamic[bl] << g - solver.Sigma_Hartree[bl]

    solver.G_iw << inverse( inverse(solver.G0_iw) - solver.Sigma_iw )

    
def tail_fit(Sigma_iw, 
             fit_min_n=None, fit_max_n=None, fit_min_w=None, fit_max_w=None, 
             fit_max_moment=None, fit_known_moments=None):

    # Define default tail quantities
    if fit_min_w is not None: fit_min_n = int(0.5*(fit_min_w*Sigma_iw.mesh.beta/np.pi - 1.0))
    if fit_max_w is not None: fit_max_n = int(0.5*(fit_max_w*Sigma_iw.mesh.beta/np.pi - 1.0))
    if fit_min_n is None: fit_min_n = int(0.8*len(Sigma_iw.mesh)/2)
    if fit_max_n is None: fit_max_n = int(len(Sigma_iw.mesh)/2)
    if fit_max_moment is None: fit_max_moment = 3

    if fit_known_moments is None:
        fit_known_moments = {}
        for name, sig in Sigma_iw:
            shape = [0] + list(sig.target_shape)
            fit_known_moments[name] = np.zeros(shape, dtype=complex) # no known moments

    # Now fit the tails of Sigma_iw and replace the high frequency part with the tail expansion
    for name, sig in Sigma_iw:

        tail, err = fit_hermitian_tail_on_window(
            sig,
            n_min = fit_min_n,
            n_max = fit_max_n,
            known_moments = fit_known_moments[name],
            # set max number of pts used in fit larger than mesh size, to use all data in fit
            n_tail_max = 10 * len(sig.mesh), 
            expansion_order = fit_max_moment
            )
        
        replace_by_tail(sig, tail, n_min=fit_min_n)        

    return Sigma_iw


def extract_Uijkl_from_h_int(h_int, gf_struct):
    """
    Return Uijkl tensor in TRIQS's notation from a Coulomb many-body operator h_int
    """
    from triqs.operators.util.extractors import extract_U_dict2, dict_to_matrix

    U_dd = dict_to_matrix(extract_U_dict2(h_int), gf_struct=gf_struct)
    n_orb = U_dd.shape[0]//2

    # extract Uijij (inter- and intra-orbital Coulomb) and Uijji (Hund's coupling) terms
    # a) For static impurity problem, Us are the static screened interactions
    # b) For dynamic impurity problem, Us are the bare interactions
    Uijij = U_dd[:n_orb, n_orb:2*n_orb]
    Uijji = Uijij - U_dd[:n_orb, 0:n_orb]
    
    # construct full Uijkl tensor for static interaction
    orb_range = range(0, n_orb)
    Uijkl = np.zeros((n_orb, n_orb, n_orb, n_orb), dtype=complex)

    # assuming Uijji = Uiijj
    for i, j, k, l in product(range(n_orb), repeat=4):
        if i == j == k == l:  # Uiiii
            Uijkl[i, i, i, i] = Uijij[i, j]
        elif i == k and j == l:  # Uijij
            Uijkl[i, j, i, j] = Uijij[i, j]
        elif i == l and j == k:  # Uijji
            Uijkl[i, j, j, i] = Uijji[i, j]
        elif i == j and k == l:  # Uiijj
            Uijkl[i, i, k, k] = Uijji[i, k]
    
    return Uijkl


def extract_screen_matrix_from_D0_tau(blk2_D0_tau, gf_struct):
    n_color = 0
    for _, blk_dim in gf_struct:
        n_color += blk_dim

    block_name = []
    index_in_block = []
    for color in range(n_color):
        block_name.append(find_block_name(color, gf_struct))
        index_in_block.append(find_index_in_block(color, gf_struct))

    mesh = blk2_D0_tau[block_name[0], block_name[0]].mesh
    D0_tau = Gf(mesh=mesh, target_shape=(n_color, n_color))
    for c1 in range(n_color):
        for c2 in range(n_color):
            D0_tau.data[:, c1, c2] = blk2_D0_tau[block_name[c1], block_name[c2]].data[:, index_in_block[c1], index_in_block[c2]]

    w0_mesh = MeshImFreq(beta = D0_tau.mesh.beta, statistic="Boson", n_iw = 1)
    D0_iw = Gf(mesh=w0_mesh, target_shape=D0_tau.target_shape)
    D0_iw.set_from_fourier(D0_tau, make_zero_tail(D0_iw, n_moments=2))
    D0_w0 = D0_iw.data[0].real

    n_orb = D0_w0.shape[0]//2
    return D0_w0[:n_orb,:n_orb]


def find_block_name(color, gf_struct):
    bl, colors_so_far = 0, 0
    for blk_name, blk_dim in gf_struct:
        colors_so_far += blk_dim
        if color < colors_so_far: 
            return blk_name
        bl+=1
    raise ValueError(f"Color index {color} out of bounds for gf_struct of total size {colors_so_far}")
    return "none"


def find_index_in_block(color, gf_struct):
    colors_so_far = 0
    for blk_name, blk_dim in gf_struct:
        colors_so_far += blk_dim
        if color < colors_so_far:
            return color - (colors_so_far - blk_dim)
    raise ValueError(f"Color index {color} out of bounds for gf_struct of total size {colors_so_far}")
    return 0


def find_orbital_index(color, gf_struct):
    n_color = 0
    for _, blk_dim in gf_struct:
        n_color += blk_dim
    n_orb = n_color // 2
    
    colors_so_far = 0
    o_up, o_dn = 0, 0
    for blk_name, blk_dim in gf_struct:
        colors_so_far += blk_dim
        if blk_name[:2] == "up":
            if color < colors_so_far:
                return o_up + ( color - (colors_so_far - blk_dim) )
            else:
                o_up += blk_dim
            assert o_up <= n_orb, f"Spin up orbital index {o_up} excceds band range {n_orb}"
        else:
            if color < colors_so_far:
                return o_dn + ( color - (colors_so_far - blk_dim) )
            else:
                o_dn += blk_dim
            assert o_dn <= n_orb, f"Spin down orbital index {o_dn} excceds band range {n_orb}"    

    raise ValueError(f"Color index {color} out of bounds for gf_struct of total size {colors_so_far}")
    return 0




