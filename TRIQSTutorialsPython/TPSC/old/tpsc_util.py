




from copy import deepcopy

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fftshift
from scipy.optimize import brentq, minimize_scalar

from fourier_transforms import fourier2k, fourier2x, fourier2w, HighFrequencyTail, fourier2wf


class ImFreqPropagator:
    """
    Propagator in Matsubara frequency representation. May contain information on leading high-frequency tails.
    """
    def __init__(self, data, frequency_axis=None, tail=None, dropped_frequencies=0):
        self.data = data
        self.frequency_axis = frequency_axis
        self.tail = tail
        self.dropped_frequencies = dropped_frequencies

    @property
    def real(self):
        return ImFreqPropagator(self.data.real, self.frequency_axis, self.tail.real, self.dropped_frequencies)

    @property
    def imag(self):
        return ImFreqPropagator(self.data.imag, self.frequency_axis, self.tail.imag, self.dropped_frequencies)

    def __getitem__(self, item):
        # figure out whether the frequency axis survives indexing and where it ends up
        freq_selected = False
        dropped_freqs = self.dropped_frequencies
        if np.isscalar(item):
            if self.frequency_axis == 0:
                freq_selected = True
            else:
                freq_axis = self.frequency_axis - 1
        elif isinstance(item, tuple):
            assert np.all([isinstance(i, (int, slice)) for i in item])
            if len(item) > self.frequency_axis:
                freq_index = item[self.frequency_axis]
                if isinstance(freq_index, int) or freq_index.stop is not None:
                    freq_selected = True
                else:
                    assert freq_index.step is None  # cannot sum tail with stride
                    if freq_index.start is not None:
                        dropped_freqs += freq_index.start
            if not freq_selected:
                freq_axis = self.frequency_axis
                for i in item[:self.frequency_axis]:
                    if isinstance(i, int):
                        freq_axis -= 1
        else:
            # working out reliably whether the frequency axis is affected would be somewhat tedious
            raise NotImplementedError("fancy indexing is not supported")
        # after selecting specific frequencies we don't need the tail any more
        if freq_selected:
            return self.data[item]
        # remove frequency axis from index tuple
        try:
            tail_index = item[:self.frequency_axis] + item[self.frequency_axis+1:]
        except TypeError:
            assert np.isscalar(item)
            tail_index = item
        # index data and tail coefficients
        return ImFreqPropagator(self.data[item], freq_axis, self.tail[tail_index], dropped_freqs)

    # def frequency_index(self, i):
    #     idx = [slice(None) for _ in np.shape(self)]
    #     idx[self.frequency_axis] = i
    #     return tuple(idx)

    def sum(self, axis=None, tau_sign=-1, tail_eps=None):
        """Sum over given axis or axes. Include analytical frequency sum from tails if the frequency axis is summed.
        :param axis: Scalar or tuple identifying axes to be summed. Defaults to None (sum all axes).
        :param tau_sign: Sign of regularization parameter \tau, see HighFrequencyTail.frequency_sum.
        :param tail_eps: Tolerance for significance of tail coefficients, see HighFrequencyTail.frequency_sum.
        """
        nfreqs = self.data.shape[self.frequency_axis] + self.dropped_frequencies
        assert nfreqs % 2 == 0
        data = self.data.sum(axis=axis)
        tailaxis = self.map_axis_to_tail(axis)
        tail = self.tail.sum(axis=tailaxis)
        if axis is not None and np.shape(tailaxis) == np.shape(axis):
            # no frequency sum => need to construct propagator object with tail
            freq_axis = self.frequency_axis
            if np.isscalar(axis) and axis < self.frequency_axis:
                freq_axis -= 1
            elif np.ndim(axis) > 0:
                for a in axis:
                    if a < self.frequency_axis:
                        freq_axis -= 1
            assert freq_axis >= 0
            return ImFreqPropagator(data, frequency_axis=freq_axis, tail=tail)
        # do frequency sum and return the data
        return data + tail.frequency_sum(minfreq=nfreqs//2, tau_sign=tau_sign, eps=tail_eps)

    def map_axis_to_tail(self, axis):
        """
        Map axis index or axis index tuple to corresponding axis indices for the tail, which does not have a frequency
        axis.
        """
        assert axis != self.frequency_axis
        # None means all axes in both cases
        if axis is None:
            return None
        # single axis index
        if np.isscalar(axis):
            # we don't support negative indexes at the moment (would make the following comparisons more complicated)
            assert axis >= 0
            # axes before frequency axis are unchanged
            if axis < self.frequency_axis:
                return axis
            # axes afterwards are shifted
            if axis > self.frequency_axis:
                return axis - 1
            # tail does not have an index corresponding to frequency axis
            return ()
        # axis tuple: map each index
        return tuple(self.map_axis_to_tail(a) for a in axis if a != self.frequency_axis)

    def __neg__(self):
        return ImFreqPropagator(-self.data, self.frequency_axis, -self.tail, self.dropped_frequencies)

    def __add__(self, other):
        try:
            # add two propagators
            assert self.frequency_axis == other.frequency_axis
            assert self.dropped_frequencies == other.dropped_frequencies
            return ImFreqPropagator(self.data + other.data, self.frequency_axis, self.tail + other.tail,
                                    self.dropped_frequencies)
        except AttributeError:
            # add scalar to propagator
            return ImFreqPropagator(self.data + other, self.frequency_axis, self.tail + other, self.dropped_frequencies)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        try:
            # subtract two propagators
            assert self.frequency_axis == other.frequency_axis
            assert self.dropped_frequencies == other.dropped_frequencies
            return ImFreqPropagator(self.data - other.data, self.frequency_axis, self.tail - other.tail,
                                    self.dropped_frequencies)
        except AttributeError:
            # subtract scalar from propagator
            return ImFreqPropagator(self.data - other, self.frequency_axis, self.tail - other, self.dropped_frequencies)

    def __rsub__(self, other):
        return -(self - other)

    def __mul__(self, other):
        try:
            # multiply two propagators
            assert self.frequency_axis == other.frequency_axis
            assert self.dropped_frequencies == other.dropped_frequencies
            return ImFreqPropagator(self.data * other.data, self.frequency_axis, self.tail * other.tail,
                                    self.dropped_frequencies)
        except AttributeError:
            # multiply propagator with scalar
            return ImFreqPropagator(self.data * other, self.frequency_axis, self.tail * other, self.dropped_frequencies)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        try:
            # divide two propagators
            assert self.frequency_axis == other.frequency_axis
            assert self.dropped_frequencies == other.dropped_frequencies
            return ImFreqPropagator(self.data / other.data, self.frequency_axis, self.tail / other.tail,
                                    self.dropped_frequencies)
        except AttributeError:
            # divide propagator by scalar
            return ImFreqPropagator(self.data / other, self.frequency_axis, self.tail / other, self.dropped_frequencies)

    def __rtruediv__(self, other):
        # division of two propagators should be handled by __truediv__
        assert not hasattr(other, 'frequency_axis')
        # divide scalar by propagator
        return ImFreqPropagator(other / self.data, self.frequency_axis, other / self.tail, self.dropped_frequencies)


def bz_mesh(l, basis=2.*np.pi*np.eye(3), shift=True):
    """k mesh over the Brillouin zone"""
    d = len(basis)
    assert np.shape(basis) == (d,d)
    grid, step = np.linspace(0, 1, l, endpoint=False, retstep=True)
    if shift:
        grid += step / 2.
    k = np.meshgrid(*(d*(grid,)), indexing='ij')
    k = np.tensordot(np.transpose(basis), k, 1)
    # k = np.tensordot(basis, k, 1)
    return np.rollaxis(k, 0, d+1)


def g_on_mesh(gfunc, tgrid, l, s=0, basis=2.*np.pi*np.eye(3), shiftk=False):
    k = bz_mesh(l, basis, shift=shiftk)
    d = len(basis)
    assert k.shape[-1] == d
    tgrid = np.reshape(tgrid, (-1,)+d*(1,))
    return gfunc(tgrid, k[np.newaxis], s=s)



def density(mdl, l, s=slice(None), a=None):
    """Calculate total (default) or spin density (if s is an int indicating the species)."""
    k = bz_mesh(l, mdl.kbasis)
    d = np.shape(k)[-1]
    if a is None:
        nk = mdl.n0k(k, s)
    else:
        assert mdl.basissize > 1
        nk = mdl.n0k(k, s, a)
    n = 1./(l**d) * nk.sum()
    # Normalize density with number of sublattice sites we have summed over
    if a is None:
        n /= mdl.basissize
    else:
        n /= np.size(np.arange(mdl.basissize)[a])
    return n

def find_mu_for_n(mdl, l, n, s=None):
    """Determine chemical potential(s) \\mu_s such that the density is n.
    
    mdl:    The model.
    l:      Linear system size L.
    n:      Target density. May be a scalar or an array. 
        If n is an array, s must be an array of the same length and specify 
        the corresponding spin indices. If n is a scalar and s an array, we 
        determine a spin-independent chemical potential such that the total 
        density is n. Otherwise n is taken to be the target spin density.
    s:      Spin(s) for which the chemical potential is to be determined. The
        default is [0,1], i.e. all spins.
    """
    if s is None:
        s = list(range(mdl.spins))
    if np.ndim(n) > 0:
        return np.array([find_mu_for_n(mdl,l,nn,ss) for (nn,ss) in zip(n,s)])
    mdl = deepcopy(mdl)
    def n_of_mu(mu):
        mdl.mu[s] = mu
        return density(mdl,l,s)
    if n < 0 or n > np.size(s):
        raise RuntimeError('invalid density: {}'.format(n))
    a = -np.mean(mdl.bandwidth(s))
    b =  np.mean(mdl.bandwidth(s))
    while n_of_mu(a) > n:   a *= 2
    while n_of_mu(b) < n:   b *= 2
    try:
        mu = brentq(lambda m: n_of_mu(m)-n,a,b,disp=True) #, full_output=True)
        # print 'brentq found mu(n={},s={})={} after {} iterations.'.format(n,s,mu,res.iterations)
    except RuntimeError as e:
        print('brentq failed:', e)
        print('Falling back on minimize_scalar.')
        res = minimize_scalar(lambda m: abs(n_of_mu(m)-n))
        if abs(n_of_mu(res.x)-n) > 1e-6:
            raise RuntimeError('minimize_scalar failed to find mu(n={},s={}): '+
                'n(mu={}) = {} != {}.'.format(n,s,res.x,n_of_mu(res.x),n)) #+res.message)
        mu = res.x
    return mu
        

def ph_bubble(model, taugrid, l, s1=0, s2=0, shiftk=True, fitspline=True, include_tail=True):
    """Compute the particle-hole bubble \\Pi(q) = -\\int dk G_{s1}(k+q) G_{s2}(k) with free propagators.

        :returns \\Pi(q): indices [[a,b,] \\nu, q_x, q_y [, q_z]]
    """
    # non-interacting Green's functions
    gtp = g_on_mesh(model.g0taukp,  taugrid, l, s=s1, basis= model.kbasis, shiftk=shiftk)
    gtm = g_on_mesh(model.g0taukm, -taugrid, l, s=s2, basis=model.kbasis, shiftk=shiftk)
    taxis = 0
    if model.basissize > 1:
        assert np.ndim(gtm) == 3 + len(model.kbasis)
        taxis = 2
        gtm = np.rollaxis(gtm, 1, 0)  # transpose sublattice indices of second propagator: \\chi_{ab} = -G_{ab} G_{ba}
    kaxes = tuple(range(taxis+1, gtp.ndim))
    assert len(kaxes) == len(model.kbasis)
    gtp = fourier2x(gtp, axes=kaxes, overwrite_x=True)
#    gtm = fourier2x(gtm, axes=kaxes, overwrite_x=True)
    assert np.all(np.isreal(gtm))
    gtm = fourier2x(gtm, axes=kaxes, overwrite_x=True).conj()

    # particle-hole bubble
    #  pi(x) = - gtp(x) * gtm(-x)
    pi = gtp
    pi *= gtm
    del gtp, gtm
    pi *= -1
    pi = fourier2k(pi, kaxes, overwrite_x=True)
    pi = fourier2w(pi, model.beta, axis=taxis, overwrite_x=True, fitspline=fitspline, return_tail=include_tail)
    if include_tail:
        pi = ImFreqPropagator(pi[0], frequency_axis=taxis, tail=HighFrequencyTail(pi[1], model.beta, bosonic=True))
    return pi





def pp_bubble(model, taugrid, l, s1, s2, shiftk=True):
    """Compute the particle-particle bubble \\Pi(q) = \\int dk G_{s1}(k+q) G_{s2}(-k) with free propagators."""
    # non-interacting Green's functions
    gtp = g_on_mesh(model.g0taukm, -taugrid, l, s=s1, basis=model.kbasis, shiftk=shiftk)
    gtm = g_on_mesh(model.g0taukm, -taugrid, l, s=s2, basis=model.kbasis, shiftk=shiftk)
    taxis = 0
    if model.basissize > 1:
        assert np.ndim(gtm) == 3 + len(model.kbasis)
        taxis = 2
    kaxes = tuple(range(taxis+1, gtp.ndim))
    assert len(kaxes) == len(model.kbasis)
    gtp = fourier2x(gtp, axes=kaxes, overwrite_x=True)
    gtm = fourier2x(gtm, axes=kaxes, overwrite_x=True)
    # particle-particle bubble
    #  pi(x) = gtp(x) * gtm(x)
    pi = gtp
    pi *= gtm
    del gtp, gtm
    pi = fourier2k(pi, kaxes, overwrite_x=True)
    pi = fourier2w(pi, model.beta, axis=taxis, overwrite_x=True)
    return pi





def plotFreqDependence(w, f, color=None, marker=('o','v'), ls=('-','--'), label=None):
    if color is None or mpl.is_string_like(color) or np.shape(color) in [(3,),(4,)]:
        color  = (color,color)
    if mpl.is_string_like(marker):  marker = (marker,marker)
    if mpl.is_string_like(ls):      ls     = (ls,ls)
    nf = len(w)
    assert len(f) == nf
    w = fftshift(w)
    f = fftshift(f)
    plt.plot(w,np.real(f),'o',color=color[0],ls=ls[0],marker=marker[0],label=label)
    if not np.allclose(np.imag(f),0):
        plt.plot(w,np.imag(f),'v',color=color[0],ls=ls[1],marker=marker[1])
    plt.show()

