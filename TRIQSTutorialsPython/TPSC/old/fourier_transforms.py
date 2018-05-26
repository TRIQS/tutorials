import numpy as np
from numpy.fft import ifftshift
from scipy import interpolate
from scipy.fftpack import fftn, ifftn, fft, ifft
from scipy.special import polygamma, factorial


def freqIndices(n):
    if n % 2 == 0:  return ifftshift(np.arange(-n//2    ,n//2    ))
    else:           return ifftshift(np.arange(-(n-1)//2,(n+1)//2))


def bosonFreqs(n,beta):
    return 2*freqIndices(n)*np.pi/beta


def fermionFreqs(n,beta):
    return (2*freqIndices(n)+1)*np.pi/beta


def broadcast1D(a,ndim,axis):
    """broadcast 1d array a to Nd along given axis"""
    shp = np.ones(ndim,dtype=np.int)
    shp[axis] = -1
    return np.reshape(a,shp)


def fourier2k(fx, axes=(-2,-1), overwrite_x=False):
    return fftn(fx, axes=axes, overwrite_x=overwrite_x)


def fourier2x(fk, axes=(-2,-1), overwrite_x=False):
    return ifftn(fk, axes=axes, overwrite_x=overwrite_x)


def fourier2t(fw, beta, axis=0, overwrite_x=False):
    ft = fft(fw, axis=axis, overwrite_x=overwrite_x)
    ft *= 1./beta
    return ft


def fourier2w_fft(ft, beta, axis=0, overwrite_x=False):
    r"""
    Compute the Fourier transform of a bosonic quantity from imaginary time to Matsubara frequencies using FFT.

    :param ft   d-dimensional array containing the input imaginary-time data in the range [0, \beta].
    :param beta Inverse temperature \beta.
    :param axis Index of the imaginary-time axis.
    :param overwrite_x  If True the input array ft may be overwritten.

    The main work is accomplished by an FFT, but the boundary terms need to be treated more carefully: We want to
    compute
        \hat{f}(w) = \int_0^\beta dt f(t) exp(i w_n t)  (1)
    with
        w_n = 2 \pi n T, n = -N/2, ... N/2.
    Since ft contains data for the time points
        t_m = m \beta / (N-1), m = 0, N-1,
    a straightforward FFT of the N values in ft would yield data for frequencies 2 \pi n T (N-1)/N = w_n (N-1)/N.
    Instead, we compute the FFT of the N-1 values f(t_0), ... f(t_{N-2}), yielding data for the correct frequencies
    w_{-N/2+1}, ... w_{N/2-1}, and compute the missing frequency w_{N/2} separately.
    At this point, we have computed the left-rectangular approximation to the integral (1). We improve on this and
    restore some symmetries by adding the contribution from the boundaries
        \beta/2N [f(t_{N-1} exp(i w_n \beta) - f(t_0) exp(i w_n 0)] = \beta/2N [f(t_{N-1} - f(t_0)]
    such that we end up with the trapezoidal rule
        \hat{f}_n = \beta/N (1/2 [f(0) - f(\beta)] + \sum_{m=1}^{N-2} f(t_m) exp(i w_n t_m) .
    """
    def tidx(i):
        idx = [slice(None) for _ in np.shape(ft)]
        idx[axis] = i
        return tuple(idx)

    nt = np.shape(ft)[axis]
    boundary = 0.5 * (ft[tidx(slice(-1, None))] - ft[tidx(slice(0, 1))])
    if overwrite_x:
        boundary = boundary.copy()
    fw = np.empty_like(ft, dtype=np.complex)
    fw[tidx(slice(0, -1))] = ifft(ft[tidx(slice(0, -1))], axis=axis, overwrite_x=overwrite_x)
    fw[tidx(slice(nt//2+1, None))] = fw[tidx(slice(nt//2, -1))]
    expwn2t = broadcast1D(np.exp(1j*2*np.pi*(nt//2)*np.arange(nt-1)/(nt-1)), np.ndim(ft), axis)
    fw[tidx(nt//2)] = np.mean((ft[tidx(slice(0, -1))] * expwn2t), axis)
    fw *= (nt - 1)
    fw += boundary
    fw *= beta / nt
    return fw


def fourier2w(ft, beta, axis=0, overwrite_x=False, fitspline=True, nf=None, return_tail=False):
    r"""
    Compute the Fourier transform of a bosonic quantity from imaginary time to Matsubara frequencies.

    :param ft   d-dimensional array containing the input imaginary-time data in the range [0, \beta].
    :param beta Inverse temperature \beta.
    :param axis Index of the imaginary-time axis.
    :param overwrite_x  If True the input array ft may be overwritten.
    :param fitspline    (True) Perform Fourier transform by fitting a cubic spline to the input data in order to
                        determine the leading coefficients of the high-frequency tail.
                        (False) Compute Fourier transform with a straight-forward FFT, only treating the 1/iw tail
                        analytically.
    :param nf   Number of frequencies to compute the output for. Default to the number of input time points.
    :param return_tail: If true, the function returns the fit coefficients of the high-frequency tails in addition to
                the Fourier transform.
    :return If return_tail==False, the Fourier transform of ft, which has the same shape as ft unless a different number
                of frequencies is required via the nf argument. If return_tail==True a tuple of the Fourier transform
                and the high-frequency tail coefficients.

    Note: We could improve the accuracy for functions where we know the derivatives at the boundaries. This is currently
    not supported by scipy.interpolate, however.
    This feature may appear in scipy 0.18.0: https://github.com/scipy/scipy/pull/5734
    """
    assert fitspline or not return_tail  # we can only calculate the tail if we do the fitting
    if not fitspline:
        return fourier2w_fft(ft, beta, axis, overwrite_x)
    ft = np.real_if_close(ft)
    if nf is None:
        nf = np.shape(ft)[axis]
    w = bosonFreqs(nf, beta)
    nt = np.shape(ft)[axis]
    t = np.linspace(0, beta, nt)
    assert np.allclose(np.exp(1j*w*t[0]), 1) and np.allclose(np.exp(1j*w*t[-1]), 1)
    expwt0, expwtn = 1, 1
    expwdt = np.exp(1j * w * (t[1] - t[0]))
    assert w[0] == 0.
    w[0] = np.nan  # avoid divide by zero warnings, zeroth frequency is calculated separately anyway
    expiwt = np.exp(1j * w.reshape(-1,1) * t[:-1])
    ft = np.rollaxis(ft, axis, np.ndim(ft))
    fw = np.empty(np.shape(ft)[:-1] + (nf,), dtype=np.complex)
    if return_tail:
        tail = np.zeros((4,) + np.shape(ft)[:-1], dtype=np.result_type(ft, expwt0, expwtn))

    # fit time dependence with cubic spline
    spline = interpolate.CubicSpline(t, ft, axis=-1, extrapolate=False)
    valt0, valtn = spline(t[0]), spline(t[-1])
    derivt0, derivtn = ([spline(tt, n) for n in range(1, 3)] for tt in (t[0], t[-1]))
    deriv3 = spline.derivative(3)(t[:-1])
    integral = spline.integrate(0., beta)

    # compute high-frequency tails from spline derivatives at the boundaries
    c1 = -(expwt0*valt0 - expwtn*valtn)
    c2 = (expwt0*derivt0[0] - expwtn*derivtn[0])
    c3 = -(expwt0*derivt0[1] - expwtn*derivtn[1])
    fw[:] = c1[..., np.newaxis] / (1j*w) + c2[..., np.newaxis] / (1j*w)**2 + c3[..., np.newaxis] / (1j*w)**3
    if return_tail:
        tail[1] = c1
        tail[2] = c2
        tail[3] = c3

    # obtain remaining parts from Fourier transform of third derivative
    fw += np.tensordot(deriv3, expiwt, (-1, 1)) * (1 - expwdt) / (1j * w)**4
    # The zero-frequency part is just the time integral
    fw[..., 0] = integral
    # restore position of frequency axis
    fw = np.rollaxis(fw, -1, axis)
    if return_tail:
        return fw, tail
    return fw



def fourier2wf(ft, beta, axis=0, overwrite_x=False, fitspline=True, nf=None, usefft=False):
    r"""
    Compute the Fourier transform of a fermionic quantity from imaginary time to Matsubara frequencies.

    :param ft   d-dimensional array containing the input imaginary-time data in the range [0, \beta].
    :param beta Inverse temperature \beta.
    :param axis Index of the imaginary-time axis.
    :param overwrite_x  If True the input array ft may be overwritten.
    :param fitspline    True: Perform Fourier transform by fitting a cubic spline to the input data in order to
                        determine the leading coefficients of the high-frequency tail.
                        False: Compute Fourier transform with a straight-forward FFT, only treating the 1/iw tail
                        analytically.
    :param nf   Number of frequencies to compute the output for. Default to the number of input time points.
    :param usefft       Use an FFT for transforming the 3rd derivative of the fitted spline.
                        FIXME: This produces rather inaccurate results for the lowest frequencies.

    Note: We could improve the accuracy for functions where we know the derivatives at the boundaries. This is currently
    not supported by scipy.interpolate, however. This feature may appear in scipy 0.18.0.
    """
    if not fitspline:
        assert nf is None
        return fourier2wf_fft(ft, beta, axis, overwrite_x)
    if nf is None:
        nf = np.shape(ft)[axis]
    w = fermionFreqs(nf, beta)
    nt = np.shape(ft)[axis]
    t = np.linspace(0, beta, nt)
    expwt0 = np.exp(1j*w*t[0])
    expwtn = np.exp(1j*w*t[-1])
    expwdt = np.exp(1j * w * (t[1] - t[0]))
    if usefft:
        # I guess the problem with using ifft for transforming the 3rd spline derivative is that our tau grid does not
        # match what ifft expects: We have t[-1]==beta at the end of the interval, whereas ifft expects
        #  t[-1]==(N-1)/N*beta.
        # The most pronounced deviations in the Fourier transform of G appear at energies far below the Fermi surface,
        # where G(tau) is strongly peaked at tau=beta.
        assert False
        phase = np.exp(1j*np.pi*np.arange(nt)/nt)
    else:
        expiwt = np.exp(1j * w.reshape(-1,1) * t[:-1])
    ft = np.rollaxis(ft, axis, np.ndim(ft))
    fw = np.empty(np.shape(ft)[:-1] + (nf,), dtype=np.complex)
    for idx in np.ndindex(*ft.shape[:-1]):
        # fit time dependence with cubic spline
        spline = interpolate.InterpolatedUnivariateSpline(t, ft[idx].real, k=3, ext='raise')
        valt0, valtn = spline(t[0]), spline(t[-1])
        derivt0, derivtn = spline.derivatives(t[0]), spline.derivatives(t[-1])
        deriv3 = spline.derivative(3)(t[:-1])
        # scipy.interpolate ignores the imaginary part, so we need to treat that separately if present
        if np.iscomplexobj(ft):
            spline = interpolate.InterpolatedUnivariateSpline(t, ft[idx].imag, k=3, ext='raise')
            valt0 += 1j * spline(t[0])
            valtn += 1j * spline(t[-1])
            derivt0 += 1j * spline.derivatives(t[0])
            derivtn += 1j * spline.derivatives(t[-1])
            deriv3 += 1j * spline.derivative(3)(t[:-1])
        # compute high-frequency tails from spline derivatives at the boundaries
        fw[idx] = -(expwt0*valt0 - expwtn*valtn) / (1j*w)
        fw[idx] +=  (expwt0*derivt0[1] - expwtn*derivtn[1]) / (1j*w)**2
        fw[idx] += -(expwt0*derivt0[2] - expwtn*derivtn[2]) / (1j*w)**3
        # obtain remaining parts from Fourier transform of third derivative
        if usefft:
            remainder = phase * deriv3
            fw[idx] += nt * ifft(remainder) * (1 - expwdt) / (1j * w)**4
        else:
            fw[idx] += np.sum(expiwt * deriv3, axis=1) * (1 - expwdt) / (1j * w)**4
    fw = np.rollaxis(fw, -1, axis)
    return fw


def fourier2tf(fw, beta, axis=0, overwrite_x=False):
    nf = np.shape(fw)[axis]
    w     = broadcast1D( fermionFreqs(nf,beta),              np.ndim(fw), axis )
    phase = broadcast1D( np.exp(-1j*np.arange(nf)*np.pi/nf), np.ndim(fw), axis )
    if not overwrite_x:
        fw = fw.copy()
    fw -= 1./(1j*w)  # remove tail
    ft = fft(fw, axis=axis, overwrite_x=overwrite_x)
    ft *= 1./beta * phase
    ft -= 0.5  # add tail back
    return ft


class HighFrequencyTail:
    def __init__(self, coefficients, beta, bosonic):
        self.coefficients = np.asanyarray(coefficients)
        self.beta = beta
        self.bosonic = bosonic

    @property
    def real(self):
        real = HighFrequencyTail(self.coefficients.copy(), self.beta, self.bosonic)
        real.coefficients[1::2] = 0  # zero out coefficients of odd orders in 1/(i w)
        return real

    @property
    def imag(self):
        imag = HighFrequencyTail(self.coefficients.copy(), self.beta, self.bosonic)
        imag.coefficients[0::2] = 0  # zero out coefficients of even orders in 1/(i w)
        return imag

    def __getitem__(self, item):
        if isinstance(item, tuple):
            return HighFrequencyTail(self.coefficients[(slice(None),) + item], self.beta, self.bosonic)
        return HighFrequencyTail(self.coefficients[:, item], self.beta, self.bosonic)

    def __neg__(self):
        return HighFrequencyTail(-self.coefficients, self.beta, self.bosonic)

    def __add__(self, other):
        if np.isscalar(other):
            # a scalar is added to the constant tail ~1/(iw)^0
            result = self.coefficients.copy()
            result[0] += other
            return HighFrequencyTail(result, self.beta, self.bosonic)
        try:
            # tail coefficients add up
            num_coeffs = min(len(self.coefficients), len(other.coefficients))
            assert self.coefficients[:num_coeffs].shape == other.coefficients[:num_coeffs].shape
            return HighFrequencyTail(self.coefficients[:num_coeffs] + other.coefficients[:num_coeffs],
                                     self.beta, self.bosonic)
        except AttributeError:
            raise TypeError

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if np.isscalar(other):
            # a scalar is added to the constant tail ~1/(iw)^0
            result = self.coefficients.copy()
            result[0] -= other
            return HighFrequencyTail(result, self.beta, self.bosonic)
        try:
            # tail coefficients add up
            num_coeffs = min(len(self.coefficients), len(other.coefficients))
            assert self.coefficients[:num_coeffs].shape == other.coefficients[:num_coeffs].shape
            return HighFrequencyTail(self.coefficients[:num_coeffs] - other.coefficients[:num_coeffs], self.beta,
                                     self.bosonic)
        except AttributeError:
            raise TypeError

    def __rsub__(self, other):
        return -(self - other)

    def __mul__(self, other):
        if np.isscalar(other):
            # a scalar  multiplies all tail coefficients
            return HighFrequencyTail(other * self.coefficients, self.beta, self.bosonic)
        try:
            num_coeffs = min(len(self.coefficients), len(other.coefficients))
            assert self.coefficients[:num_coeffs].shape == other.coefficients[:num_coeffs].shape
            dtype = np.result_type(self.coefficients, other.coefficients)
            result = np.zeros_like(self.coefficients[:num_coeffs], dtype=dtype)
            for n in range(num_coeffs):
                for p in range(n+1):
                    result[n] += self.coefficients[p] * other.coefficients[n-p]
            return HighFrequencyTail(result, self.beta, self.bosonic)
        except AttributeError:
            raise TypeError

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if np.isscalar(other):
            # a scalar  divides all tail coefficients
            return HighFrequencyTail(self.coefficients / other, self.beta, self.bosonic)
        try:
            # truncate to 3rd order because we haven't implemented higher-order expressions below
            num_coeffs = min(4, len(self.coefficients), len(other.coefficients))
            a = self.coefficients[:num_coeffs]
            b = other.coefficients[:num_coeffs]
            assert a.shape == b.shape
            dtype = np.result_type(self.coefficients, other.coefficients)
            result = np.zeros_like(a, dtype=dtype)

            # if the denominator does not have a constant part, we need to multiply numerator and denominator with (iw)
            finite_b0 = (b[0] != 0)
            if np.any(~finite_b0):
                if np.any(a[0, ~finite_b0] != 0):
                    raise ValueError('high-frequency tail diverges')
                shifted_a = HighFrequencyTail(a[1:, ~finite_b0], self.beta, self.bosonic)
                shifted_b = HighFrequencyTail(b[1:, ~finite_b0], self.beta, self.bosonic)
                shifted_div = (shifted_a / shifted_b).coefficients
                result[:-1, ~finite_b0] = shifted_div

            a, b = a[:, finite_b0], b[:, finite_b0]
            if num_coeffs > 0:
                result[0, finite_b0] = a[0] / b[0]
            if num_coeffs > 1:
                result[1, finite_b0] = (a[1]*b[0] - a[0]*b[1]) / b[0]**2
            if num_coeffs > 2:
                result[2, finite_b0] = (a[2]*b[0]**2 - a[1]*b[0]*b[1] + a[0]*b[1]**2 - a[0]*b[0]*b[2]) / b[0]**3
            if num_coeffs > 3:
                result[3, finite_b0] = (a[3]*b[0]**3 - a[2]*b[0]**2*b[1] + a[1]*b[0]*b[1]**2 - a[0]*b[1]**3
                                        - a[1]*b[0]**2*b[2] + 2*a[0]*b[0]*b[1]*b[2] - a[0]*b[0]**2*b[3]) / b[0]**4
            return HighFrequencyTail(result, self.beta, self.bosonic)
        except AttributeError:
            raise TypeError

    def __rtruediv__(self, other):
        # division of tails should be handled by __truediv__
        assert not hasattr(other, 'coefficients')

        # truncate to 3rd order because we haven't implemented higher-order expressions below
        num_coeffs = min(4, len(self.coefficients))
        a = other
        b = self.coefficients[:num_coeffs]
        result = np.zeros_like(b, dtype=np.result_type(a, b))

        # if the denominator does not have a constant part the fraction diverges for iw->\infty
        if np.any(b[0] == 0):
            raise ValueError('high-frequency tail diverges')

        if num_coeffs > 0:
            result[0] = a / b[0]
        if num_coeffs > 1:
            result[1] = -a * b[1] / b[0]**2
        if num_coeffs > 2:
            result[2] = a * (b[1]**2 - b[0] * b[2]) / b[0]**3
        if num_coeffs > 3:
            result[3] = a * (2 * b[0] * b[1] * b[2] - b[1]**3 - b[0]**2 * b[3]) / b[0]**4
        return HighFrequencyTail(result, self.beta, self.bosonic)

    @staticmethod
    def distribution_function(beta, xi, bosonic, tau_sign=-1):
        r"""
        Calculate the generalized boson or fermion distribution function for inverse temperature beta and energy xi.

        With
            n_\eta(\xi) = 1 / [exp(\beta \xi) - \eta]
        and \eta = +1 (-1) for bosons (fermions) we find for the generalized distribution function
            G_\eta(\xi, \tau->0) = T \sum_n exp(-i \omega_n \tau) / [i \omega_n - \xi]
                                 = -\eta n_\eta(\xi)  for \tau=0-,
                                 = -\eta n_\eta(\xi) - 1  for \tau=0+.

        :param beta Inverse temperature.
        :param xi Energy \xi.
        :param bosonic Bose (True) or Fermi (False) distribution.
        :param tau_sign Sign of the regularization \tau -> 0^{+/-}.
        :param bosonic Sum over bosonic (True) or fermionic (False) Matsubara frequencies.
        :return Value of the distribution function; same type as (beta * xi)
        """
        if bosonic:
            g = -1. / (np.exp(beta*xi) - 1.)
        else:
            g = 1. / (np.exp(beta*xi) + 1.)
        if tau_sign > 0:
            g -= 1
        return g

    def frequency_sum(self, minfreq, tau_sign=-1, eps=None):
        r"""Perform the sum of the tail over frequencies +/- (minfreq...\infty)
        :param minfreq Index of the lowest positive Matsubara frequency to be summed.
        :param tau_sign Sign of the regularization \tau=0^{+/-} in f(\tau=0+) = \sum_w exp(-i w \tau) f(w); this is
                    relevant for the 1/iw tail.
        :param eps Tolerance for deciding whether coefficient of 1/iw tail is significant.
        :return Result of the tail summed over the specified frequencies.
        """
        if eps is None:
            eps = 1e-12
        if self.coefficients.ndim == 1:
            # add dummy axis to avoid special indexing cases
            self.coefficients.shape += 1,
            is_scalar = True
        else:
            is_scalar = False
        result = np.zeros(self.coefficients.shape[1:], dtype=np.complex)
        result[self.coefficients[0] != 0] = np.inf  # infinite number of constants

        # Tails with slow 1/(iw) part need to be done carefully:
        # We calculate
        #   A/T G(\xi) = \sum_n A / (iw - \xi)
        # with A and \xi determined by the first- and second-order tails.
        has_slow_tail = np.where(np.abs(self.coefficients[1]) > eps + np.sqrt(eps)*np.abs(self.coefficients[2]))
        a = self.coefficients[1][has_slow_tail]
        xi = self.coefficients[2][has_slow_tail] / a
        result[has_slow_tail] = self.beta * a * self.distribution_function(self.beta, xi, self.bosonic, tau_sign)

        # Subtract low frequencies |w| < sumfreq that we were not supposed to sum from result
        if self.bosonic:
            w = bosonFreqs(2*minfreq, self.beta)
        else:
            w = fermionFreqs(2*minfreq, self.beta)
        result[has_slow_tail] -= np.sum(a[..., np.newaxis] / (1j*w - xi[..., np.newaxis]), axis=-1)

        # Remove expansion of the part we've taken care of
        #   A / (iw - \xi) = \sum_{k=1}^\infty A \xi^{k-1} / (iw)^k
        # from coefficients.
        coefficients = self.coefficients.copy()
        for k, c in enumerate(coefficients[1:]):
            c[has_slow_tail] -= a * xi**k

        # Analytically sum fast-decaying tails ~1/(iw)^k with k>=2
        if self.bosonic:
            for k in range(2, len(coefficients)):
                # \sum_{m=n}^\infty 1/(iw)^k =
                tail = 1j**k * self.beta**k / (2 * np.pi)**k * polygamma(k - 1, minfreq) / factorial(k - 1)
                # for even powers positive and negative frequencies add up, for odd k they cancel
                if k % 2 == 0:
                    tail *= 2
                else:
                    tail = 0.
                # bosonic frequency ranges are not symmetric: (w_{-minfreq}, ..., w_{minfreq-1}) are summed explicitly
                # so we need to remove w_{-minfreq} from tail
                assert w[minfreq] == -2*minfreq*np.pi/self.beta
                tail -= 1./(1j*w[minfreq])**k
                result += coefficients[k] * tail
        else:
            for k in range(2, len(coefficients)):
                # for odd powers positive and negative frequencies cancel
                if k % 2 != 0:
                    continue
                # \sum_{m=n}^\infty 1/(iw)^k =
                tail = 1j**k * self.beta**k / (2 * np.pi)**k * polygamma(k - 1, minfreq + 0.5) / factorial(k - 1)
                # factor 2 due to sum of positive and negative frequencies
                result += 2 * coefficients[k] * tail

        if is_scalar:
            # remove dummy axis
            assert self.coefficients.shape[1:] == (1,) and result.shape == (1,)
            self.coefficients = self.coefficients[..., 0]
            result = result[0]
        return result

    def sum(self, axis=None):
        r"""Perform sum over given axes (default: all axes)."""
        # indices to coefficients array are larger by one because first index denotes order in 1/iw
        if axis is None:
            axis = tuple(range(1, self.coefficients.ndim))
        elif np.isscalar(axis):
            axis += 1
        else:
            axis = tuple(a+1 for a in axis)
        coefficients = self.coefficients.sum(axis=axis)
        tail_sum = HighFrequencyTail(coefficients, self.beta, self.bosonic)
        return tail_sum
