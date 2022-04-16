import numpy as np


# Waveform approximants
# =====================

WfWrapper = {} # collect all the wvf wrappers


# Example of waveform wrapper for PyROQ
# A wrapper must be a class with these attributes/methods
# -------------------------------------------------------

class ZeroWf:
    def __init__(self,
                 approximant,
                 waveform_params = {}):
        self.approximant = approximant
        self.waveform_params = waveform_params
    def generate_waveform(self, waveform_params, deltaF, f_min, f_max):
        freq = np.arange(f_min,f_max,deltaF)
        hp = np.zeros(len(freq))
        hc = np.zeros(len(freq))
        return hp, hc


# LAL
# ---

try:
    
    # LAL imports
    import lal, lalsimulation
    from lal.lal import PC_SI as LAL_PC_SI, MSUN_SI as LAL_MSUN_SI

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
        def __init__(self,
                     approximant,
                     waveform_params = {}):

            self.approximant = approximant

            if waveform_params:
                self.waveform_params = waveform_params
            else:
                self.waveform_params = lal.CreateDict()                
                
        def generate_waveform(self, p, deltaF, f_min, f_max):

            # Update baseline waveform_params with p
            # incomplete, see
            # https://github.com/gwastro/pycbc/blob/master/pycbc/waveform/waveform.py#L77
            if p['lambda1'] is not None:
                lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(self.waveform_params, p['lambda1'])
            if p['lambda2'] is not None:
                lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(self.waveform_params, p['lambda2'])
            
            [plus, cross] =
            lalsimulation.SimInspiralChooseFDWaveform(p['m1']*LAL_MSUN_SI,
                                                      p['m2']*LAL_MSUN_SI,
                                                      p['s1x'], p['s1y'], p['s1z'],
                                                      p['s2x'], p['s2y'], p['s2z'],
                                                      p['distance'],
                                                      p['iota'],
                                                      p['phiRef'],
                                                      0,  # float(p['long_asc_nodes'])
                                                      p['ecc'],
                                                      0,  # float(p['mean_per_ano'])
                                                      deltaF,
                                                      f_min,
                                                      f_max,
                                                      0,
                                                      self.waveform_params,
                                                      self.approximant)
            hp = plus.data.data
            hc = cross.data.data
            hp = hp[np.int(f_min/deltaF):np.int(f_max/deltaF)]
            hc = hc[np.int(f_min/deltaF):np.int(f_max/deltaF)]
            return hp, hc

    
    # Add a wrapper for each approximant
    for a in approximants:
        WfWrapper[a] = LALWf

except ModuleNotFoundError:
    print('LAL module not found.')

    # Set the constants, they might be needed
    # https://lscsoft.docs.ligo.org/lalsuite/lal/group___l_a_l_constants__h.html
    LAL_PC_SI   = 3.085677581491367278913937957796471611e16
    LAL_MSUN_SI = 1.988409902147041637325262574352366540e30


# TEOBResumS
# ----------
    
try:
    # TEOBResumS imports
    import EOBRun_module

    # Add the approximants that can be called
    approximants = []
    approximants.append('teobresums-giotto')

    # Helpers
    TEOBResumS_domain = {'TD':0,'FD':1}
    TEOBResumS_spins  = {'nospin':0,'aligned':1,'precessing':2}

    # TEOBResumS wrapper
    class WfTEOBResumS:
        def __init__(self,
                     approximant,
                     waveform_params = {}):
            
            self.approximant = approximant

            if waveform_params:
                self.waveform_params = waveform_params
            else:
                self.waveform_params = self.set_parameters()

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

            p['use_geometric_units'] = 0      # Output quantities in geometric units. Default = 1
            p['interp_uniform_grid'] = 2      # Interpolate mode by mode on a uniform grid. Default = 0 (no interpolation)
            # print('CHECKME: why interp?')
            # print('CHECKME: finish review of all TEOB options')

            p['output_hpc'         ] = 0
            p['output_multipoles'  ] = 0

            p['use_spins'] = TEOBResumS_spins['aligned'] 
            p['use_mode_lm'] = [1] # [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 13 ]
            p['domain'] = TEOBResumS_domain['FD']

            return p

        def JBJF(self,hp,hc,dt):
            """
            Fourier transform of TD wfv
            """
            hptilde = np.fft.rfft(hp) * dt 
            hctilde = np.fft.rfft(-hc) * dt 
            return hptilde, hctilde
  
        def generate_waveform(self, p, deltaF, f_min, f_max):

            # Eccentric binaries are not supported
            if(abs(p['ecc') > 1e-12):
               raise ValueError("Eccentricity is not supported, but eccentricity={} was passed.".format(p['ecc']))

            # Impose the correct convention on masses
            m1,m2 = p['m1'],p['m2']
            q = p['m1']/p['m2']
            lambda1,lambda2 = p['lambda1'],p['lambda2']
            s1x,s1y,s1z = p['s1x'], p['s1y'], p['s1z']
            s2x,s2y,s2z = p['s2x'], p['s2y'], p['s2z']

            if q < 1. :
               m1,m2       = m2,m1
               q           = 1./q
               t1,t2,t3    = s1x,s1y,s1z 
               s1x,s1y,s1z = s2x,s2y,s2z
               s2x,s2y,s2z = t1,t2,t3
               lambda1,lambda2 = lambda2,lambda1

            # System parameters
            p['M'                  ] = m1+m2
            p['q'                  ] = q
            p['LambdaAl2'          ] = lambda1
            p['LambdaBl2'          ] = lambda2
            if p['use_spins'] == TEOBResumS_spins['precessing']:
                p['chi1x'          ] = s1x
                p['chi1y'          ] = s1y
                p['chi1z'          ] = s1z
                p['chi2x'          ] = s2x
                p['chi2y'          ] = s2y
                p['chi2z'          ] = s2z
            else:
                p['chi1'           ] = s1z
                p['chi2'           ] = s2z
            p['distance'           ] = p['distance']/(LAL_PC_SI*1e6) # Mpc
            p['inclination'        ] = p['iota']
            p['coalescence_angle'  ] = p['phiRef']

            # Generator parameters
            p['srate'              ] = f_max*2  # srate at which to interpolate. Default = 4096.
            p['srate_interp'       ] = f_max*2  # srate at which to interpolate. Default = 4096.
            p['initial_frequency'  ] = f_min  # in Hz if use_geometric_units = 0, else in geometric units
            p['df'                 ] = deltaF

            # Update wave_params with p
            self.wave_params.update(p)
               
            if p['domain'] == TEOBResumS_domain['TD']:
                t, hp, hc = EOBRun_module.EOBRunPy(self.wave_params)
                Hptilde, Hctilde = JBJF(hp,hc,t[1]-t[0])
            else:
                f, rhplus, ihplus, rhcross, ihcross = EOBRun_module.EOBRunPy(self.wave_params)

            # Adapt len to PyROQ frequency axis conventions
            hp, hc = rhplus[:-1]-1j*ihplus[:-1], rhcross[:-1]-1j*ihcross[:-1]
            return hp, hc

    
    # Add a wrapper for each approximant
    for a in approximants:
        WfWrapper[a] = WfTEOBResumS

except ModuleNotFoundError:
    print('TEOBResumS module not found')


# MLGW-BNS
# --------
    
try:
    # MLGW imports
    ##TODO!!!

    # Add the approximants that can be called
    approximants = []
    approximants.append('mlgw-bns')

    # MLGW-BNS wrapper
    class WfMLGW:
        def __init__(self,
                     approximant,
                     waveform_params = {}):
               
            self.approximant = approximant
            self.waveform_params = waveform_params
  
        def generate_waveform(self, p, deltaF, f_min, f_max):
            
            # eccentric binaries are not supported
            if(abs(p['ecc']) > 1e-12):
                raise ValueError("Eccentricity is not supported, but eccentricity={} was passed.".format(p['ecc']))

            # Impose the correct convention on masses
            m1,m2 = p['m1'],p['m2']
            q = p['m1']/p['m2']
            lambda1,lambda2 = p['lambda1'],p['lambda2']
            s1x,s1y,s1z = p['s1x'], p['s1y'], p['s1z']
            s2x,s2y,s2z = p['s2x'], p['s2y'], p['s2z']

            if q < 1. :
               m1,m2       = m2,m1
               q           = 1./q
               s1z,s2z     = s2z,s1z
               lambda1,lambda2 = lambda2,lambda1
               
            # Precessing spins are not supported
            if((abs(s1x) > 1e-6) or (abs(s1y) > 1e-6)):
                raise ValueError("Precession is not supported, but (spin1x, spin1y)=({},{}) were passed.".format(s1x, s1y))
            if((abs(s2x) > 1e-6) or (abs(s2y) > 1e-6)):
                raise ValueError("Precession is not supported, but (spin2x, spin2y)=({},{}) were passed.".format(s2x, s2y))
            
            # Call it
            model       = Model.default() ##TODO here you can use self.approximant to call any MLGW-BNS Model
            frequencies = np.arange(f_min, f_max, step=deltaF)
            params      = ParametersWithExtrinsic(p['q'],
                                                  p['lambda1'],
                                                  p['lambda2'],
                                                  p['s1z'],
                                                  p['s2z'],
                                                  p['distance']/(LAL_PC_SI*1e6),
                                                  p['iota'],
                                                  p['m1']+p['m2'],
                                                  reference_phase=p['phiRef'])
            hp, hc      = model.predict(frequencies, params)
            return hp, hc

    # Add a wrapper for each approximant
    for a in approximants:
        WfWrapper[a] = WfMLGW
        
except ModuleNotFoundError:
    print('mlgw-bns module not found')

