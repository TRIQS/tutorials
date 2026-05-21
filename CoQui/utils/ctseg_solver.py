import numpy as np
from triqs.gfs import MeshImFreq, Gf, BlockGf, Block2Gf, MeshDLRImFreq, MeshImFreq, make_gf_from_fourier, fit_hermitian_tail, make_hermitian
from triqs.gfs import make_gf_dlr, make_gf_imfreq, make_gf_dlr_imfreq, make_gf_imtime, fit_gf_dlr, inverse, iOmega_n
from triqs.gfs.dlr_crm_dyson_solver import minimize_dyson
from triqs.gfs.tools import make_zero_tail
from triqs.operators import c_dag, c
from triqs.operators.util.U_matrix import reduce_4index_to_2index
from triqs.operators.util.extractors import block_matrix_from_op

from triqs_ctseg import Solver
import ctseg_utils


class SmartList(list):
    def __init__(self, **kwargs): #self._data = { key : val for key, val in kwargs.items() }
        super().__init__(kwargs)


    def __getitem__(self, key):
        if isinstance(key, int): return self.__values()[key]
        #if isinstance(key, str): return self._data[key]
        if isinstance(key, str): return super().__getitem__(key)

    def __getattr__(self, key):
        try: return self._data[key]
        except KeyError: raise AttributeError(f"SmartList has no property {key}")

    def __iter__(self): return iter(self.__values())

    def __repr__(self): return repr(self._data)
    __str__ = __repr__

    def to_vector(self): return self.__values()

class SolverResults(dict):
    def __init__(self, **kwargs):
        super().__init__(kwargs)
    def __getattr__(self, key): 
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"SolverResults has no property {key}")
    def __repr__(self):
        return "\n".join([f"{key:<12}" for key in self.keys() ])
        
    __str__ = __repr__

def matrix_to_many_body_operator(h_loc_matrix, gf_struct):
    h_loc0_mat = {block : h_loc_matrix[ibl].real for ibl, (block, _) in enumerate(gf_struct) }
    c_dag_vec  = {block : np.matrix([[c_dag(block, o) for o in range(bl_size)]]) for block, bl_size in gf_struct }
    c_vec      = {block : np.matrix([[c(block, o) for o in range(bl_size)]]) for block, bl_size in gf_struct }
    return sum(c_dag_vec[block]*h_loc0_mat[block]*c_vec[block] for block, bl_size in gf_struct)[0,0]


def solve_dynamic_full_mesh(Delta_iw, h_loc0, D0_iw, h_int, **solver_interface_params):
    """
    Solve the impurity with dynamic interactions 

    Delta_iw : triqs.BlockGf
    h_loc0: triqs.operator
    D0_iw : triqs.Block2Gf 
    h_int: triqs.operator
    """
    gf_struct = [(bl, gf.target_shape[0]) for (bl, gf) in Delta_iw]
    beta  = Delta_iw.mesh.beta
    n_iw = len(Delta_iw.mesh) // 2
    solver_interface_params.pop('n_iw', None)
    n_tau = solver_interface_params.pop('n_tau', 10001)
    n_tau_bosonic = solver_interface_params.pop('n_tau_bosonic', n_tau)
    
    S = Solver(gf_struct=gf_struct, beta=beta, n_tau=n_tau, n_tau_bosonic=n_tau_bosonic)

    # initialization for post proccessing 
    post_proc_params = {}
    post_proc_params['perform_tail_fit'] = solver_interface_params.pop('perform_tail_fit', False)
    post_proc_params['fit_max_moment']   = solver_interface_params.pop('fit_max_moment', 3)
    post_proc_params['fit_min_w']        = solver_interface_params.pop('fit_min_w', None)
    post_proc_params['fit_max_w']        = solver_interface_params.pop('fit_max_w', None)
    post_proc_params['fit_min_n']        = solver_interface_params.pop('fit_min_n', None)
    post_proc_params['fit_max_n']        = solver_interface_params.pop('fit_max_n', None)
    post_proc_params['analytic_hf']      = solver_interface_params.pop('analytic_hf', False)
    S.n_iw, S.beta, S.gf_struct = n_iw, beta, gf_struct     # useful and neccessary for post-processing
    S.h_int = h_int
    S.h_loc0_mat = block_matrix_from_op(h_loc0, gf_struct)

    # prepare Delta_tau
    for block, delta in S.Delta_tau:
        S.Delta_tau[block] << make_gf_from_fourier(
            Delta_iw[block],                                          # Δ(iω)
            S.Delta_tau[block].mesh,                                  # time mesh
            fit_hermitian_tail(Delta_iw[block], make_zero_tail(Delta_iw[block], 1))[0] # tail
        )
        
    # prepare D0_tau
    for name1, name2 in D0_iw.indices:
        S.D0_tau[name1, name2] << make_gf_from_fourier(
            D0_iw[name1, name2],                                          # D0(iω)
            S.D0_tau[name1, name2].mesh,                                  # time mesh
            fit_hermitian_tail(D0_iw[name1, name2], make_zero_tail(D0_iw[name1, name2], 1))[0] # tail
        )
        
    # call solver
    solver_interface_params['measure_densities'] = True
    solver_interface_params['measure_F_tau']     = True
    solver_interface_params['measure_nn_tau']    = True
    S.solve(h_loc0=h_loc0, h_int=h_int, **solver_interface_params)
    
    # post process
    ctseg_utils.post_process(S, **post_proc_params)
    
    return SolverResults(# required
                         G_iw  = S.G_iw, 
                         Sigma_Hartree = list(S.Sigma_Hartree.values()),
                         Sigma_dynamic = S.Sigma_dynamic, 
                         Pi_iw = S.Pi_iw, 
                         W_dynamic = S.W_iw, 

                         # optional
                         G_tau = S.results.G_tau,
                         Sigma_iw = S.Sigma_iw, 
                         orbital_occupations = S.results.densities, 
                         average_order = S.results.average_order_Delta,
                         average_sign = S.results.average_sign
    )
    

def solve_dynamic_imp(Delta_iw, h_loc0, U_iw, h_int, **solver_params):
    return solve_dynamic_full_mesh(Delta_iw, h_loc0, U_iw, h_int, **solver_params)
    
