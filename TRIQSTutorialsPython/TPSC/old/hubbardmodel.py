import numpy as np
import sys
SPINS = 2

#Classe contenant les parametres du probleme ainsi que quelques fonctions importantes
#Notamment les fonctions de Green en temps imaginaire .

class HubbardModel(object):
    def __init__(self,parms):
        parms = parms.copy()
        self.spins = SPINS
        self.basissize = 1
        self.dims = int(parms.pop('DIMENSIONS',2))
        if not self.dims in [2,3]:
            raise RuntimeError('only 2 or 3 dimensions supported')
        self.kbasis = 2*np.pi*np.eye(self.dims)
        self.l = np.inf*np.ones(self.dims)
        if 'LINEAR_SYSTEM_SIZE' in parms:
            self.l[:] = parms.pop('LINEAR_SYSTEM_SIZE')
        for d,c in enumerate(['X','Y','Z'][:self.dims]):
            if 'LINEAR_SYSTEM_SIZE'+c in parms:
                self.l[d] = parms.pop('LINEAR_SYSTEM_SIZE'+c)
        if 'BETA' in parms: self.beta = float(parms.pop('BETA'))
        else:               self.beta = float(1./parms.pop('T'))
        if 'HUBBARDU' in parms: self.u    = float(parms.pop('HUBBARDU'))
        else:                   self.u    = float(parms.pop('U'))
        self.t    = np.zeros((SPINS,self.dims))
        self.tP   = 0.
        self.tPP   = 0.
        self.mu   = np.zeros(SPINS)
        # hopping amplitudes for all spins and directions
        if 'HOPPING' in parms:
            self.t[:,:] = parms.pop('HOPPING')
        elif 't' in parms:
            self.t[:,:] = parms.pop('t')
        elif 'HOPPINGX' in parms:
            for d,c in enumerate(['X','Y','Z'][:self.dims]):
                self.t[:,d] = parms.pop('HOPPING'+c)
        else:
            for d,c in enumerate(['X','Y','Z'][:self.dims]):
                for s in range(SPINS):
                    self.t[s,d] = parms.pop('HOPPING'+c+str(s))
        if 'HOPPINGXY' in parms:
            self.tP = float(parms.pop('HOPPINGXY'))
        elif 'tP' in parms:
            self.tP = float(parms.pop('tP'))
        if 'HOPPING2' in parms:
            self.tPP = float(parms.pop('HOPPING2'))
        elif 'tPP' in parms:
            self.tPP = float(parms.pop('tPP'))
        # chemical potential for all spins
        if 'MU' in parms:
            self.mu[:]   = parms.pop('MU')
        else:
            for s in range(SPINS):
                self.mu[s] = parms.pop('MU'+str(s))
        # 
        self.kmin = -np.pi*np.ones(self.dims)
        self.kmax =  np.pi*np.ones(self.dims)
        if parms:
            print("warning: Unused parameters:", parms.keys())
    
    def __repr__(self):
        return "HubbardModel(%s)" % repr(self.parms())
    
    def __eq__(self,other):
        eps = 1e-10
        return np.abs(self.beta-other.beta) < eps and np.abs(self.u-other.u) < eps \
            and np.all(np.abs(self.t-other.t) < eps) and np.all(np.abs(self.mu-other.mu) < eps) \
            and self.dims==other.dims
    
    def __ne__(self,other):
        return not (self==other)

    def parms(self):
        p = {}
        p['DIMENSIONS'] = self.dims
        if np.any(np.isfinite(self.l)):
            if np.all(self.l == self.l[0]):
                p['LINEAR_SYSTEM_SIZE'] = self.l[0]
            else:
                p['LINEAR_SYSTEM_SIZE'] = self.l
        p['BETA'] = self.beta
        p['U'] = self.u
        p['t'] = self.t
        p['tP'] = self.tP
        p['tPP'] = self.tPP
        p['MU'] = self.mu
        return p

    def bandwidth(self,s=slice(SPINS)):
        t = self.t.mean(axis=1)[s]
        if self.tP == 0:
            return 4*self.dims*np.abs(t)
        ratio = t / self.tP
        if self.dims == 2:
            return np.where(np.abs(ratio) >= 2, 8*np.abs(t), 4*np.abs(t) + 8*np.abs(self.tP))
        elif self.dims == 3:
            return np.where(np.abs(ratio) >= 4, 12*np.abs(t), 8*np.abs(t) + 16*np.abs(self.tP))

    def dispersion(self,k,s=slice(SPINS)):
        k = np.asanyarray(k)
        assert np.shape(k)[-1] == self.dims
        cosk = np.cos(k)
        cos2k = np.cos(2*k)
        eps = -2 * np.tensordot(self.t[s], cosk, (-1,-1))
        if self.dims == 2:
            epsp = -4 * self.tP * cosk[...,0] * cosk[...,1]
            epspp = -2*self.tPP*(cos2k[...,0]+cos2k[...,1])
        else:  # the following expression is valid for any dimension, except in 2D it would count cos(kx)*cos(ky) twice
            epsp = -4 * self.tP * np.sum(cosk * np.roll(cosk, 1, axis=-1), axis=-1)
        return eps + epsp + epspp
    
    def xi(self,k,s):
        '''Dispersion shifted by chemical potential: $\\xi_s(k) = \epsilon_s(k) - \\mu_s$'''
        eps = self.dispersion(k,s)
        mu = self.mu[s]
        if np.ndim(mu) > 0: mu.shape = mu.shape + tuple(1 for _ in np.shape(eps)[1:])
        return eps - mu
        
    def g0taukp(self,tau,k,s=slice(SPINS)):
        '''G_s(\tau>0,k)'''
        assert np.all(tau >= 0) and np.all(tau <= self.beta)
        xi = self.xi(k,s)
        shift = (xi < 0) * xi*self.beta
        return -np.exp(-xi*tau+shift) / (1 + np.exp(-xi*self.beta+2*shift))

    def g0taukm(self,tau,k,s=slice(SPINS)):
        '''G_s(\tau<0,k)'''
        assert np.all(tau <= 0) and np.all(tau >= -self.beta)
        xi = self.xi(k,s)
        shift = (xi > 0) * xi*self.beta
        return np.exp(-xi*tau-shift) / (1 + np.exp( xi*self.beta-2*shift))

    def g0tauk(self,tau,k,s=slice(SPINS)):
        '''G_s(\tau,k)'''
        while np.any(tau <= -self.beta):  tau[tau <= -self.beta] += 2.*self.beta
        while np.any(tau  >  self.beta):  tau[tau  >  self.beta] -= 2.*self.beta
        return np.where( tau > 0, self.gktaup(k,tau,s), self.gktaum(k,tau,s) )
    
    def g0wk(self,n,k,s=slice(SPINS)):
        '''G_s(i \omega_n,k)'''
        iw = 1j*(2*n+1)*np.pi/self.beta
        xi = self.xi(k,s)
        return 1 / (iw - xi)
    
    def n0k(self,k,s=slice(SPINS)):
        return self.g0taukm(0,k,s)
        
    def gg0kw(self,n,k1,k2,s1=0,s2=1):
        g1 = self.g0wk(n   ,k1,s1)
        g2 = self.g0wk(-n-1,k2,s2)
        return g1*g2
