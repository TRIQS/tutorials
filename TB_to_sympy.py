# importing the dependencies
from itertools import product as itp
from pythtb import *
from triqs.lattice.tight_binding import TBLattice
import warnings
import numpy as np
import matplotlib.pyplot as plt

def TB_to_sympy(TBL, analytical = True, precision = 6):
    r"""
    returns the analytical form of the momentum space hamiltonian of the tight-binding model 
    from a tight-binding lattice object by utilizing Fourier series
    
    Parameters
    ----------
    TBL: triqs TBLattice object
        triqs tight binding object
    analytical: boolean, default = True
        a boolean which will cause the function will return an analytical Hamiltonian, when true, and 
        an numerical Hamiltonian otherwise
    precision: integer, default = 6
        specifies the precision of the floating point amplitudes. The default precision is 6 but the user
        can decrease this parameter for better recognition of similar hopping amplitudes, particularly
        for symmetrical hoppings across the crystal lattice
    
    Returns
    -------
    Hk: NumPy array
        the Hamiltonian of the tight-binding model in momentum space. It can be output in either numerical
        form (Hk_numerical) or reduced analytical form (Hk) based on the user's choice. The default output
        is the reduced analytical form. The numerical form depends solely on the k-space vector components
        while the analytical form takes into account both the k-space vector components and the lattice
        vectors

    """

    import sympy as sp
    
    # imaginary number
    I = sp.I

    # matrix from axis directions in momentum space
    kx, ky, kz = sp.symbols("kx ky kz", real = True)
    k_space_matrix = sp.Matrix([kx, ky, kz])

    # symbolic dot product representation between lattice unit vectors
    # and momentum space matrix
    a1k, a2k, a3k = sp.symbols("a1k a2k a3k", real = True)
    lattice = sp.Matrix([a1k, a2k, a3k])

    # units contains the displacement vectors
    # hops contains details about hopping of electrons such as orbital
    # and hopping amplitude
    if TBL.units.shape == (2, 2):
        TBL_units = np.eye(3)
        TBL_units[:2, :2] = TBL.units
        TBL_hops = {key + (0,): val for key, val in TBL.hoppings.items()}
    elif TBL.units.shape == (3,3):
        TBL_units = TBL.units
        TBL_hops = TBL.hoppings
    # raises error for when the dimensions of the tb object is neither 2D nor 3D
    else:
        raise ValueError("This format of the tight-binding model is not implemented for this function.")
   
    # number of orbitals involved in the unit cell
    num_orb = TBL.n_orbitals

    # maximum hopping distances of electrons in each direction
    max_x, max_y, max_z = list(np.max(np.array(list(TBL_hops.keys())), axis = 0))

    # number of cells involved in the hopping of electrons in each direction
    num_cells_x, num_cells_y, num_cells_z = [2 * max_coord + 1 for max_coord in [max_x, max_y, max_z]]
    
    # real-space Hamiltonian
    Hrij = np.zeros((num_cells_x, num_cells_y, num_cells_z, num_orb, num_orb), dtype = sp.exp)

    # looping through hopping parameters of electrons involved in inter-orbital hoppings
    for key, hopping in TBL_hops.items():
        rx, ry, rz = key
        hopping = np.around(hopping, precision)
        Hrij[rx + max_x, ry + max_y, rz + max_z] = hopping

    # basis of the exponential term in calculation of Hk
    Hexp = np.empty_like(Hrij, dtype = sp.exp)

    # perform Fourier transform
    for xi, yi, zi in itp(range(num_cells_x), range(num_cells_y), range(num_cells_z)):
        coefficients = np.array([xi - max_x, yi - max_y, zi - max_z])
        r = lattice.dot(coefficients)
        eikr = sp.exp(-I * r)
        Hexp[xi, yi, zi, :, :] = eikr

    # summation over all real space axes
    Hk = np.sum(Hrij * Hexp, axis = (0, 1, 2))
    
    # rewriting exponential terms in Hamiltonian expression in terms of cosine
    for i, j in itp(range(num_orb), repeat = 2):
        Hk[i, j] = Hk[i, j].rewrite(sp.cos)

    # dealing with the numerical Hamiltonian

    # convert to SymPy matrix to use substitutions method available in SymPy
    Hk_numerical = sp.Matrix(Hk)

    # obtaining individual displacement vectors
    a1 = np.around(TBL_units[0], precision)
    a2 = np.around(TBL_units[1], precision)
    a3 = np.around(TBL_units[2], precision)

    # numerical dot products between unit vectors
    # and momentum space matrix
    a1k_numerical = a1.dot(k_space_matrix)[0]
    a2k_numerical = a2.dot(k_space_matrix)[0]
    a3k_numerical = a3.dot(k_space_matrix)[0]
    
    # performing numerical dot product substitutions
    Hk_numerical = Hk_numerical.subs(a1k, a1k_numerical)
    Hk_numerical = Hk_numerical.subs(a2k, a2k_numerical)
    Hk_numerical = Hk_numerical.subs(a3k, a3k_numerical)

    # converting numerical Hamiltonian to NumPy array
    Hk_numerical = np.array(Hk_numerical)

    def _has_complex_exponential_sympy(matrix):
        """
        Checks if a NumPy array containing SymPy elements has a complex exponential element.

        Args:
            matrix (NumPy array): The input NumPy array containing SymPy elements
        
        Returns:
            bool: True if the matrix array contains a complex exponential element, False otherwise.

        """

        for sublist in matrix:
            for element in sublist:
                if element.is_complex and element.has(sp.exp):
                    return True
        return False
    
    def _is_hermitian_sympy(matrix):
        """
        Checks if a NumPy array containing SymPy elements is hermitian

        Args:
            matrix (NumPy array): The input NumPy array containing SymPy elements
        
        Returns:
            bool: True if the matrix is a hermitian, False otherwise

        """
        
        n = matrix.shape[0]
        for i in range(n):
            for j in range(n):
                if matrix[i, j] != matrix[j, i].conjugate():
                    return False
        return True
    
    # warning indicating when Hamiltonian is not hermitian
    if _is_hermitian_sympy(Hk) == False or _is_hermitian_sympy(Hk_numerical) == False:
        return warnings.warn("The resulting Hamiltonian is not hermitian.")

    # warning indicating when Hamiltonian contains complex exponential element
    if _has_complex_exponential_sympy(Hk_numerical) or _has_complex_exponential_sympy(Hk):
        return warnings.warn("""Your expression has a complex exponential. 
                                Choosing a different unit cell could make 
                                your Hamiltonian expression real.""")
    
    # returning analytical or numerical form of Hamiltonian
    if analytical:
        return Hk
    return Hk_numerical
