from triqs_maxent import PoormanMaxEnt, LogAlphaMesh, HyperbolicOmegaMesh
from triqs_maxent import InversionSigmaContinuator
from triqs.gfs import MeshImFreq, MeshImTime

def Aw_from_maxent(G, omega_min=-10, omega_max=10, alpha_min=1e-6, alpha_max=1e2, n_alpha_points=50, n_omega_points=200, error=0.04):
    omega = HyperbolicOmegaMesh(omega_min=omega_min, omega_max=omega_max, n_points=n_omega_points)
    maxent_anacont = {}
    imfreq = isinstance(G.mesh, MeshImFreq)
    for block, g in G:
        tm = PoormanMaxEnt(use_complex=True)
        if imfreq:
            tm.set_G_iw(G[block])
        else:
            tm.set_G_tau(G[block])
        tm.alpha_mesh = LogAlphaMesh(alpha_min=alpha_min, alpha_max=alpha_max, n_points=n_alpha_points)
        tm.omega = omega
        tm.set_error(error)
        maxent_anacont[block] = tm.run()
    return omega, maxent_anacont


def Sigma_w_from_maxent(Sigma_iw, omega_min=-10, omega_max=+10, n_omega_points=200, alpha_min=1e-6, alpha_max=1e2, n_alpha_points=50, error=0.04):
    omega = HyperbolicOmegaMesh(omega_min=omega_min, omega_max=omega_max, n_points=n_omega_points)
    inversion_sigma_continuator = InversionSigmaContinuator(Sigma_iw, 0.0)
    maxent_results = {}
    for block, gaux in inversion_sigma_continuator.Gaux_iw:
        tm = PoormanMaxEnt(use_complex=True)
        tm.set_G_iw(gaux)
        tm.alpha_mesh = LogAlphaMesh(alpha_min=alpha_min, alpha_max=alpha_max, n_points=n_alpha_points)
        tm.omega = omega
        tm.set_error(error)
        maxent_results[block] = tm.run()
    Aux_w = { block : result.get_A_out('LineFitAnalyzer') for block, result in maxent_results.items() }
    inversion_sigma_continuator.set_Gaux_w_from_Aaux_w(Aux_w, omega, np_interp_A=10000, np_omega=1000, w_min=omega_min, w_max=omega_max)
    return inversion_sigma_continuator.S_w