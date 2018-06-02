
""" Utilities for visualization of scalar fields in 3D (momentum space) 

Author: Hugo U. R. Strand (2018)

"""

# ----------------------------------------------------------------------

import itertools
import numpy as np

# ----------------------------------------------------------------------

from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import NearestNDInterpolator        

# ----------------------------------------------------------------------
def k_space_path(paths, num=100):

    """ High symmetry path k-vector generator.

    Input:
    paths : list of tuples of pairs of 3-vectors of k-points to
    make the path in between 
    num (optional) : number of k-vectors along each segment of the path

    Returns:
    k_vecs: ndarray.shape = (n_k, 3) with all k-vectors. 
    k_plot: ndarray.shape = (n_k) one dimensional vector for plotting
    K_plot: ndarray.shape = (n_paths) positions of the start and end of each path 
    """

    k_vecs = []

    for path in paths:
        ki, kf = path
        x = np.linspace(0., 1., num=num)[:, None]
        k_vec = (1. - x) * ki[None, :] + x * kf[None, :]

        k_vecs.append(k_vec)

    k_plot = np.linalg.norm(k_vecs[0] - k_vecs[0][0][None, :], axis=1)

    K_plot = [0.]
    for kidx, k_vec in enumerate(k_vecs[1:]):
        k_plot_new = np.linalg.norm(k_vec - k_vec[0][None, :], axis=1) + k_plot[-1]
        K_plot.append(k_plot[-1])
        k_plot = np.concatenate((k_plot, k_plot_new))

    K_plot.append(k_plot[-1])
    K_plot = np.array(K_plot)
    k_vecs = np.vstack(k_vecs)
    
    return k_vecs, k_plot, K_plot

# ----------------------------------------------------------------------
def get_relative_k_from_absolute(k_vec, units):
    k_vec_rel = np.dot(np.linalg.inv(units).T, k_vec.T).T
    return k_vec_rel

# ----------------------------------------------------------------------
def get_kidx_from_k_vec_relative(k_vec_rel, nk):
    kidx = np.array(np.round(k_vec_rel * np.array(nk)[None, :]), dtype=np.int)
    return kidx

# ----------------------------------------------------------------------
def get_k_components_from_k_vec(k_vec, nk):

    dim = 3

    shape_4 = [dim] + list(nk)
    shape_2 = [dim] + [np.prod(nk)]

    k_vec = k_vec.swapaxes(0, -1)    
    k_vec = k_vec.reshape(shape_4)

    # -- cut out values for each axis
    
    k_out = []
    for axis in xrange(dim):
        cut = [0]*dim
        cut[axis] = slice(None)
        cut = [axis] + cut
        k_out.append(k_vec[tuple(cut)])
        
    return tuple(k_out)

# ----------------------------------------------------------------------
def get_abs_k_interpolator(values, kmesh, bz, extend_bz=[0]):

    k_mat = bz.units()
    k_vec = np.array([k.value for k in bzmesh])
    
    # -- Extend with points beyond the first bz

    k_vec_ext = []
    values_ext = []
    for k_shift in itertools.product(extend_bz, repeat=3):
        k_shift = np.dot(k_mat.T, k_shift)        
        k_vec_ext.append( k_vec + k_shift[None, :] )
        values_ext.append(values)

    k_vec = np.vstack(k_vec_ext)
    values = np.hstack(values_ext)

    interp = LinearNDInterpolator(k_vec, values, fill_value=float('nan'))
    
    return interp
    
# ----------------------------------------------------------------------
def get_rel_k_interpolator(values, bzmesh, bz, nk,
                           extend_boundary=True, interpolator='regular'):

    k_mat = bz.units()
    k_vec = np.array([k.value for k in bzmesh])

    k_vec_rel = get_relative_k_from_absolute(k_vec, bz.units())
    k_idx = get_kidx_from_k_vec_relative(k_vec_rel, nk)

    kx, ky, kz = get_k_components_from_k_vec(k_vec_rel, nk)

    if extend_boundary:
        values, k_vec_rel, (kx, ky, kz) = \
            extend_data_on_boundary(values, nk)
    else:
        values = values.reshape(nk)

    # -- select interpolator type
        
    if interpolator is 'regular':
        interp = RegularGridInterpolator(
            (kx, ky, kz), values, fill_value=float('nan'), bounds_error=False)
    elif interpolator is 'nearest':
        interp = NearestNDInterpolator(k_vec_rel, values.flatten())
    elif interpolator is 'linear':
        interp = LinearNDInterpolator(k_vec_rel, values.flatten(), fill_value=float('nan'))
    elif interpolator is 'linear2D':
        interp = LinearNDInterpolator(k_vec_rel[:, :2], values.flatten(), fill_value=float('nan'))
    else:
        raise NotImplementedError
        
    return interp

# ----------------------------------------------------------------------
def extend_data_on_boundary(values, nk):

    nk = np.array(nk)

    # -- add points on the boundary

    # Add extra points in the positive index directions
    nk_ext = nk + 1 * (nk != 1) # (do not extend if we only have one k-point)
    coords = [ np.arange(0, n) for n in nk_ext ]

    Coords = np.meshgrid(*coords, indexing='ij')
    Coords_mod = [ np.mod(x, n) for x, n in zip(Coords, nk) ]

    values_ext = values.reshape(nk)[tuple([ X.flatten() for X in Coords_mod])]
    values_ext = values_ext.reshape(nk_ext)

    # -- compute kidx_ext

    k_idx_ext = np.array([ X.flatten() for X in Coords ]).T
    k_vec_rel_ext = np.array(k_idx_ext, dtype=np.float) / nk[None, :]
    kxe, kye, kze = get_k_components_from_k_vec(k_vec_rel_ext, nk_ext)

    return values_ext, k_vec_rel_ext, (kxe, kye, kze)
