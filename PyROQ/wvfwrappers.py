import numpy as np
#import h5py

# For each approximant...
WfWrapper = {} # ... collect the wvf wrappers 
NParams = {} # ... collect  the number of parameters

# Waveform approximants
# =====================

# Dummy: each wrapper must be a class like this
# ---------------------------------------------

class DummyWf:
    def __init__(self, approximant):
        self.approximant = approximant
        self.waveFlags = {}
    def generate_waveform(self, m1, m2, spin1, spin2, ecc, lambda1, lambda2, iota, phiRef, distance, deltaF, f_min, f_max):
        freq = np.arange(f_min,f_max,deltaF)
        hp = np.zeros(len(freq))
        hc = np.zeros(len(freq))
        return hp, hc

# LAL
# ---

try:
    
    # LAL imports
    import lal
    import lalsimulation
    from lal.lal import PC_SI as LAL_PC_SI
    from lal.lal import MSUN_SI as LAL_MSUN_SI

    # Add the approximants that can be called
    approximants = []
    approximants.append(lalsimulation.IMRPhenomPv2)
    approximants.append(lalsimulation.IMRPhenomPv3)
    approximants.append(lalsimulation.IMRPhenomPv3HM)
    approximants.append(lalsimulation.IMRPhenomXHM)
    approximants.append(lalsimulation.TaylorF2Ecc)
    approximants.append(lalsimulation.IMRPhenomPv2_NRTidal)
    approximants.append(lalsimulation.IMRPhenomNSBH)
    
    # LAL wrapper
    class LALWf:
        def __init__(self, approximant):
            self.approximant = approximant
            self.waveFlags = lal.CreateDict()
        def generate_waveform(self,m1, m2, spin1, spin2, ecc, lambda1, lambda2, iota, phiRef, distance, deltaF, f_min, f_max):
            lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(self.waveFlags, lambda1)
            lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(self.waveFlags, lambda2)     
            [plus, cross] = lalsimulation.SimInspiralChooseFDWaveform(test_mass1, test_mass2,
                                                                      spin1[0], spin1[1], spin1[2],
                                                                      spin2[0], spin2[1], spin2[2],
                                                                      distance, iota, phiRef,
                                                                      0, ecc, 0,
                                                                      deltaF, f_min, f_max,
                                                                      0, self.waveFlags, self.approximant)
            hp = plus.data.data
            hc = cross.data.data
            hp = hp[np.int(f_min/deltaF):np.int(f_max/deltaF)]
            hc = hc[np.int(f_min/deltaF):np.int(f_max/deltaF)]
            return hp, hc
    
    # Add a wrapper for each approximant
    for a in approximants:
        WfWrapper[a] = LALWf

    # Set NParams
    NParams[lalsimulation.IMRPhenomPv2] = 10
    NParams[lalsimulation.IMRPhenomPv3] = 10
    NParams[lalsimulation.IMRPhenomPv3HM] = 10
    NParams[lalsimulation.IMRPhenomXHM] = 10
    NParams[lalsimulation.TaylorF2Ecc] = 11
    NParams[lalsimulation.IMRPhenomPv2_NRTidal] = 12
    NParams[lalsimulation.IMRPhenomNSBH] = 12
        
except ModuleNotFoundError:
    print('LAL module not found.')

    # Set the constants, they are needed
    # https://lscsoft.docs.ligo.org/lalsuite/lal/group___l_a_l_constants__h.html
    LAL_PC_SI = 3.085677581491367278913937957796471611e16
    LAL_MSUN_SI = 1.988409902147041637325262574352366540e30


# TEOBResumS
# ----------
    
try:
    # TEOBResumS imports
    import EOBRun_module

    # Add the approximants that can be called
    # These names are just helpers to select different TEOBResumS modes
    approximants = []
    approximants.append('teobresums-giotto-TD')
    approximants.append('teobresums-giotto-FD')
    #approximants.append('teobresums-giotto-TD-HM')
    #approximants.append('teobresums-giotto-FD-HM')
    #approximants.append('teobresums-giotto-TD-prec')
    #approximants.append('teobresums-giotto-FD-prec')
    #...

    # TEOBResumS wrapper
    TEOBResumS_domain = {'TD':0,'FD':1}
    TEOBResumS_spins = {'nospin':0,'aligned':1,'precessing':2}
    class WfTEOBResumS:
        def __init__(self, approximant):
            self.approximant = approximant
            self.waveFlags = {}
            self.waveFlags = self.set_parameters()

        def modes_to_k(self,modes):
            """
            Map (l,m) -> k 
            """
            return [int(x[0]*(x[0]-1)/2 + x[1]-2) for x in modes]

        def set_parameters(self):
            """
            Utility to set EOB parameters based on the selected mode
            Uses defaults for unset parameters
            """            
            self.waveFlags['use_geometric_units'] = 0 # Output quantities in physical units
            self.waveFlags['interp_uniform_grid'] = 2 # Interpolate mode by mode on a uniform grid.

            spins = 'aligned' 
            if 'prec' in self.approximant:
                spins = 'precessing' 
            self.waveFlags['use_spins'] = TEOBResumS_spins[spins] 

            modes = [ 1 ] # List of modes to use, = self.modes_to_k([[2,2]])  
            if 'HM' in self.approximant:
                modes = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 13 ]
            self.waveFlags['use_mode_lm'] = modes

            domain  = 'FD'
            if 'TD' in self.approximant:
                domain = 'FD'    
            self.waveFlags['domain'] = TEOBResumS_domain[domain]

            return

        def JBJF(self,hp,hc,dt):
            """
            Fourier transform of TD wfv
            """
            hptilde = np.fft.rfft(hp) * dt 
            hctilde = np.fft.rfft(-hc) * dt 
            return hptilde, hctilde
  
        def generate_waveform(self, m1, m2, spin1, spin2, ecc, lambda1, lambda2, iota, phiRef, distance, deltaF, f_min, f_max):
            """
            EOB waveform
            spin[12] have 3 entries x,y,z
            """
            q = m1/m2
            if q < 1.:
                q = 1./q
                spin1,spin2 = spin2,spin1
                m1,m2 = m2,m1
                lambda1,lambda2 = lambda2,lambda1
            srate = f_max*2
            # Bring back the quantities to units compatible with TEOB 
            m1 = m1/LAL_MSUN_SI
            m2 = m2/LAL_MSUN_SI
            distance = distance/(LAL_PC_SI*1e6)
            # EOB pars to generate wvf
            self.waveFlags['M'                  ] = m1+m2
            self.waveFlags['q'                  ] = q    
            self.waveFlags['LambdaAl2'          ] = lambda1
            self.waveFlags['LambdaBl2'          ] = lambda2
            if self.waveFlags['use_spins'] == TEOBResumS_spins['precessing']:
                self.waveFlags['chi1x'] = spin1[0]
                self.waveFlags['chi1y'] = spin1[1]
                self.waveFlags['chi1z'] = spin1[2]
                self.waveFlags['chi2x'] = spin2[0]
                self.waveFlags['chi2y'] = spin2[1]
                self.waveFlags['chi2z'] = spin2[2]
            else:
                self.waveFlags['chi1'] = spin1[2] 
                self.waveFlags['chi2'] = spin2[2]
            self.waveFlags['srate_interp'] = srate  # srate at which to interpolate. Default = 4096.
            self.waveFlags['initial_frequency'] = f_min  # in Hz if use_geometric_units = 0, else in geometric units
            self.waveFlags['distance'] = distance
            self.waveFlags['inclination'] = iota
            if domain == 'TD':
                T, Hp, Hc = EOBRun_module.EOBRunPy(self.waveFlags)
                Hptilde, Hctilde = JBJF(Hp,Hc,T[1]-T[0])
            else:
                F, Hptilde, Hctilde, hlm, dyn = EOBRun_module.EOBRunPy(self.waveFlags)
            Hptilde = Hptilde[np.int(f_min/deltaF):np.int(f_max/deltaF)] 
            Hctilde = Hctilde[np.int(f_min/deltaF):np.int(f_max/deltaF)]
            return Hptilde, Hctilde

    # Add a wrapper for each approximant
    for a in approximants:
        WfWrapper[a] = WfTEOBResumS

    # Set NParams
    NParams['teobresums-giotto-FD'] = 12 # Note: these are less (spin aligned)
    NParams['teobresums-giotto-TD'] = 12 
    #NParams['teobresums-giotto-TD-HM'] = 12
    #NParams['teobresums-giotto-FD-HM'] = 12
    #NParams['teobresums-giotto-TD-prec'] = 12
    #NParams['teobresums-giotto-FD-prec'] = 12
    #...
        
except ModuleNotFoundError:
    print('TEOBResumS module not found')
    

    
