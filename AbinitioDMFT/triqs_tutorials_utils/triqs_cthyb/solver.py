import numpy as np
from triqs.gfs import MeshImFreq, MeshDLRImFreq, BlockGf, Block2Gf, Gf, make_gf_from_fourier, fit_hermitian_tail, make_hermitian
from triqs.gfs import make_gf_dlr, make_gf_imfreq, make_gf_dlr_imfreq, make_gf_imtime, fit_gf_dlr, inverse, iOmega_n
from triqs.gfs.dlr_crm_dyson_solver import minimize_dyson
from triqs.gfs.tools import make_zero_tail
from triqs.operators import c_dag, c
from triqs_cthyb import Solver

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


def solve_dlr_mesh(Delta_iw, h_loc0_bl_mat, h_int, **solver_interface_params):

    gf_struct = [(bl, gf.target_shape[0]) for (bl, gf) in Delta_iw]
    h_loc0 = matrix_to_many_body_operator(h_loc0_bl_mat, gf_struct)

    beta  = Delta_iw.mesh.beta
    w_max = Delta_iw.mesh.w_max
    eps   = Delta_iw.mesh.eps

    n_iw = solver_interface_params.pop('n_iw', 1025)
    n_tau = solver_interface_params.pop('n_tau', 10001)
    n_l   = solver_interface_params.pop('n_l', 40)
    solver_interface_params['measure_density_matrix'] = True
    solver_interface_params['use_norm_as_weight']     = True

    S = Solver(gf_struct=gf_struct, beta=beta, n_iw=n_iw, n_tau=n_tau, n_l = n_l, delta_interface=True)

    S.Delta_tau << make_gf_imtime(Delta_iw, n_tau)

    S.solve(h_loc0=h_loc0, h_int=h_int, **solver_interface_params)

    G_iw_dlr  = make_gf_dlr_imfreq(fit_gf_dlr(S.G_tau, w_max, eps))
    G0_iw_dlr = Delta_iw.copy()
    for idx, (bl, g) in enumerate(G0_iw_dlr): G0_iw_dlr[bl] << inverse(iOmega_n - h_loc0_bl_mat[idx] - Delta_iw[bl])
    Sigma_dlr, Sigma_hartree, err = minimize_dyson(G0_iw_dlr, G_iw_dlr, S.Sigma_moments)

    Sigma_iw = make_gf_imfreq(Sigma_dlr, n_iw)
    for block, g in Sigma_iw:
        g += S.Sigma_moments[block][0]

    return SolverResults(G_iw = S.G_iw,
                         G_tau = S.G_tau,
                         Sigma_Hartree = S.Sigma_Hartree.values(), #SmartList(**S.Sigma_Hartree),
                         Sigma_iw = Sigma_iw,
                         Sigma_dynamic = Sigma_dlr,
                         Sigma_iw_raw = S.Sigma_iw_raw,
                         Sigma_moments = SmartList(**S.Sigma_moments),
                         auto_corr_time =  S.auto_corr_time,
                         average_order = S.average_order,
                         average_sign = S.average_sign,
                         density_matrix = S.density_matrix,
                         h_loc_diagonalization = S.h_loc_diagonalization,
                         orbital_occupations = S.orbital_occupations,
                         performance_analysis = S.performance_analysis,
                         perturbation_order = S.perturbation_order,
                         perturbation_order_total = S.perturbation_order_total,
                        )

def solve_full_mesh(Delta_iw, h_loc0_bl_mat, h_int, **solver_interface_params):

    gf_struct = [(bl, gf.target_shape[0]) for (bl, gf) in Delta_iw]
    h_loc0 = matrix_to_many_body_operator(h_loc0_bl_mat, gf_struct)

    beta  = Delta_iw.mesh.beta
    n_iw = len(Delta_iw.mesh) // 2
    solver_interface_params.pop('n_iw', None)

    n_tau = solver_interface_params.pop('n_tau', 10001)
    n_l   = solver_interface_params.pop('n_l', 40)
    solver_interface_params['measure_density_matrix'] = True
    solver_interface_params['use_norm_as_weight']     = True

    S = Solver(gf_struct=gf_struct, beta=beta, n_iw=n_iw, n_tau=n_tau, n_l = n_l, delta_interface=True)

    for block, delta in S.Delta_tau:
        S.Delta_tau[block] << make_gf_from_fourier(
            Delta_iw[block],                                          # Δ(iω)
            S.Delta_tau[block].mesh,                                  # time mesh
            fit_hermitian_tail(Delta_iw[block], make_zero_tail(Delta_iw[block], 1))[0] # tail
            )

    S.solve(h_loc0=h_loc0, h_int=h_int, **solver_interface_params)
    Sigma_dynamic = S.Sigma_iw.copy()
    for bl, g in Sigma_dynamic: Sigma_dynamic[bl] << g - S.Sigma_Hartree[bl]

    return SolverResults(G_iw = S.G_iw,
                         G_tau = S.G_tau,
                         Sigma_Hartree = list(S.Sigma_Hartree.values()), #SmartList(**S.Sigma_Hartree),
                         Sigma_iw = S.Sigma_iw,
                         Sigma_dynamic = Sigma_dynamic,
                         Sigma_moments = S.Sigma_moments,
                         auto_corr_time =  S.auto_corr_time,
                         average_order = S.average_order,
                         average_sign = S.average_sign,
                         density_matrix = S.density_matrix,
                         h_loc_diagonalization = S.h_loc_diagonalization,
                         orbital_occupations = S.orbital_occupations,
                         performance_analysis = S.performance_analysis,
                         perturbation_order = S.perturbation_order,
                         perturbation_order_total = S.perturbation_order_total,
                        )

def solve(Delta_iw, h_loc0, h_int, **solver_params):
    if isinstance(Delta_iw.mesh, MeshImFreq):      return solve_full_mesh(Delta_iw, h_loc0, h_int, **solver_params)
    elif isinstance(Delta_iw.mesh, MeshDLRImFreq): return solve_dlr_mesh(Delta_iw, h_loc0, h_int,  **solver_params)
    else:                                          raise  NotImplemented
