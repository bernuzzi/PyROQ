# General python imports
import matplotlib, matplotlib.pyplot as plt, multiprocessing as mp, numpy as np, os, random, warnings
try:
    import configparser
except ImportError:
    import ConfigParser as configparser
from optparse import OptionParser

# Package internal import
from wvfwrappers import *

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
np.set_printoptions(linewidth=np.inf)
TermError    = ValueError("Unknown basis term requested.")
VersionError = ValueError("Unknown version requested.")
np.random.seed(150914)

# PyRQQ
# =====

#Description of the package. Printed on stdout if --help option is give.
usage="""\n\n pyroq.py --config-file config.ini\n
Package description TO BE FILLED.

Options syntax: type, default values and sections of the configuration
file where each parameter should be passed are declared below.
By convention, booleans are represented by the integers [0,1].
A dot is present at the end of each description line and is not
to be intended as part of the default value.

       **************************************************************************
       * Parameters to be passed to the [I/O] section.                          *
       **************************************************************************

               output               Output directory. Default: './'.
               verbose              Flag to activate verbose mode. Default: 1.
               show-plots           Flag to show produced plots. Default: 0.
               post-processing-only Flag to skip interpolants constructions, running post-processing tests and plots. Default: 0.
       
       **************************************************************************
       * Parameters to be passed to the [Parallel] section.                     *
       **************************************************************************

               parallel             Flag to activate parallelisation. Default: 0.
               n-processes          Number of processes on which the parallelisation is carried on. Default: 4.
       
       **************************************************************************
       * Parameters to be passed to the [Waveform_and_parametrisation] section. *
       **************************************************************************

               approximant          Waveform approximant. Can be any LAL approximant, or MISSING. Default: 'teobresums-giotto'.
               spins                Option to select spin degrees of freedom. Allowed values: ['no-spins', 'aligned', 'precessing']. Default: 'aligned'.
               tides                Flag to activate tides training. Default: 0.
               eccentricity         Flag to activate eccentricity training. Default: 0.
               mc-q-par             Flag to activate parametrisation in Mchirp and mass ratio. Default: 1.
               spin-sph             Flag to activate parametrisation in spins spherical components. Default: 0.
               f-min                Minimum of the frequency axis on which the interpolant will be constructed. Default: 20.
               f-max                Maximum of the frequency axis on which the interpolant will be constructed. Default: 1024.
               seglen               Inverse of the step of the frequency axis on which the interpolant will be constructed. Default: 4.

       **************************************************************************
       * Parameters to be passed to the [ROQ] section.                          *
       **************************************************************************
       
               n-tests-basis        Number of random validation test waveforms checked to be below tolerance before stopping adding basis elements in the interpolants construction. For testing: ~1000, for production: ~1000000. Default: 1000.
               n-tests-post         Number of random validation test waveforms checked to be below tolerance a-posteriori. Typically same as `n_tests_basis`. Default: 1000.
               error-version        DESCRIPTION MISSING. Default: 'v1'.
           
               n-basis-search-iter  Number of points for each search of a new basis element during basis construction. Typical values: 30-100 for testing; 300-2000 for production. Typically roughly comparable to the number of basis elements. Depends on complexity of waveform features, parameter space and signal length. Increasing it slows down offline construction time, but decreases number of basis elements. Default: 80.
           
               n-basis-low-lin      Lower bound on number of linear basis elements checked to give interpolation below tolerance. Default: 40.
               n-basis-hig-lin      Upper bound on number of linear basis elements checked to give interpolation below tolerance. Default: 80.
               n-basis-step-lin     Number of linear basis elements incremental step to check if the interpolation satisfies the requested tolerance. Default: 10.
               tolerance            Interpolation error threshold for linear basis elements. Default: 1e-8.
           
               n-basis-low-quad     Same as above, for quadratic basis. Default: 10.
               n-basis-hig-quad     Same as above, for quadratic basis. Usually 66% of linear basis one. Default: 20.
               n-basis-step-quad    Same as above, for quadratic basis. Default: 1.
               tolerance-quad       Same as above, for quadratic basis. Default: 1e-5.
               
       **************************************************************************
       * Parameters range and test values syntax.                               *
       **************************************************************************
       
       Allowed parameter names and units are:
       
               mc   (mc-q-par=1) : chirp mass
               q    (mc-q-par=1) : mass ratio
               m1   (mc-q-par=0) : mass object 1 [Mo]
               m2   (mc-q-par=0) : mass object 2 [Mo]
               s1s1 (spin-sph=1) : spin components object 1, spherical coords (SPECIFY)
               s1s2 (spin-sph=1) : spin components object 1, spherical coords (SPECIFY)
               s1s3 (spin-sph=1) : spin components object 1, spherical coords (SPECIFY)
               s2s1 (spin-sph=1) : spin components object 2, spherical coords (SPECIFY)
               s2s2 (spin-sph=1) : spin components object 2, spherical coords (SPECIFY)
               s2s3 (spin-sph=1) : spin components object 2, spherical coords (SPECIFY)
               s1x  (spin-sph=0) : spin components object 1, cartesian coords
               s1y  (spin-sph=0) : spin components object 1, cartesian coords
               s1z  (spin-sph=0) : spin components object 1, cartesian coords
               s2x  (spin-sph=0) : spin components object 2, cartesian coords
               s2y  (spin-sph=0) : spin components object 2, cartesian coords
               s2z  (spin-sph=0) : spin components object 2, cartesian coords
               lambda1           : tidal polarizability parameter object 1
               lambda2           : tidal polarizability parameter object 2
               ecc               : eccentricity
               iota              : inclination
               phiref            : reference phase
               
      Waveform wrappers must work with these keywords.
      Parameter ranges can be set using: par-X=value, where X can be ['min', 'max'] and par is any of the above names.

      MISSING description of distance [Mpc]

    """

# This is the training range of MLGW-BNS
default_params_ranges = {
    'mc'      : [0.9, 1.4]     ,
    'q'       : [1.0, 3.0]     ,
    's1x'     : [0.0, 0.0]     ,
    's1y'     : [0.0, 0.0]     ,
    's1z'     : [-0.5, 0.5]    ,
    's2x'     : [0.0, 0.0]     ,
    's2y'     : [0.0, 0.0]     ,
    's2z'     : [-0.5, 0.5]    ,
    'lambda1' : [5.0, 5000.0]  ,
    'lambda2' : [5.0, 5000.0]  ,
    'ecc'     : [0.0, 0.0]     ,
    'iota'    : [0.0, np.pi]   ,
    'phiref'  : [0.0, 2*np.pi] ,
}

default_test_values = {
        'mc'      : 1.3 ,
        'q'       : 2.0 ,
        's1x'     : 0.  ,
        's1y'     : 0.0 ,
        's1z'     : 0.2 ,
        's2x'     : 0.0 ,
        's2y'     : 0.0 ,
        's2z'     : 0.1 ,
        'lambda1' : 1000,
        'lambda2' : 1000,
        'ecc'     : 0.0 ,
        'iota'    : 1.9 ,
        'phiref'  : 0.6 ,
}

## Set some defaults
default_start_values = {
    'mc'      : default_params_ranges['mc'][0]     ,
    'q'       : default_params_ranges['q'][0]      ,
    's1x'     : default_params_ranges['s1x'][0]    ,
    's1y'     : default_params_ranges['s1y'][0]    ,
    's1z'     : default_params_ranges['s1z'][0]    ,
    's2x'     : default_params_ranges['s2x'][0]    ,
    's2y'     : default_params_ranges['s2y'][0]    ,
    's2z'     : default_params_ranges['s2z'][0]    ,
    'lambda1' : default_params_ranges['lambda1'][0],
    'lambda2' : default_params_ranges['lambda2'][0],
    'ecc'     : default_params_ranges['ecc'][0]    ,
    'iota'    : default_params_ranges['iota'][0]   ,
    'phiref'  : default_params_ranges['phiref'][0] ,
    }


def read_config(config_file):

    if not config_file:
        parser.print_help()
        parser.error('Please specify a config file.')
    if not os.path.exists(config_file):
        parser.error('Config file {} not found.'.format(config_file))
    Config = configparser.ConfigParser()
    Config.read(config_file)

    print('\nReading config file: {}'.format(config_file)+'.')
    print('With sections: '+str(Config.sections())+'.')
    print('\n----Input parameters----\nI\'ll be running with the following values:\n')

    # ==========================================================#
    # Initialize and read from config the ROQ input parameters. #
    # ==========================================================#

    sections  = ['I/O', 'Parallel', 'Waveform_and_parametrisation', 'ROQ']
    input_par = {}

    input_par['I/O']                           = {
                                                 'output'              : './',
                                                 'verbose'             : 1,
                                                 'show-plots'          : 0,
                                                 'post-processing-only': 0,
                                                }
    input_par['Parallel']                     = {
                                                 'parallel'            : 0,
                                                 'n-processes'         : 4,
                                                }

    input_par['Waveform_and_parametrisation'] = {
                                                 'approximant'         : 'teobresums-giotto',
                                                 'spins'               :'aligned',
                                                 'tides'               : 0,
                                                 'eccentricity'        : 0,
                                                 'mc-q-par'            : 1,
                                                 'spin-sph'            : 0,
                                                 'f-min'               : 20,
                                                 'f-max'               : 1024,
                                                 'seglen'              : 4,
                                                }
    input_par['ROQ']                          = {
                                                 'n-tests-basis'       : 1000,
                                                 'n-tests-post'        : 1000,
                                                 'error-version'       : 'v1',
                                               
                                                 'n-basis-search-iter' : 80,
                                               
                                                 'n-basis-low-lin'     : 40,
                                                 'n-basis-hig-lin'     : 80,
                                                 'n-basis-step-lin'    : 10,
                                                 'tolerance'           : 1e-8,
                                               
                                                 'n-basis-low-quad'    : 20,
                                                 'n-basis-hig-quad'    : 80,
                                                 'n-basis-step-quad'   : 10,
                                                 'tolerance-quad'      : 1e-10,
                                                }

    max_len_keyword = len('post-processing-only')
    for section in sections:
        print('[{}]\n'.format(section))
        for key in input_par[section]:
            keytype = type(input_par[section][key])
            try:
                input_par[section][key]=keytype(Config.get(section,key))
                print("{name} : {value}".format(          name=key.ljust(max_len_keyword), value=input_par[section][key]))
            except (KeyError, configparser.NoOptionError, TypeError):
                print("{name} : {value} (default)".format(name=key.ljust(max_len_keyword), value=input_par[section][key]))
        print('\n')

    # Sanity checks
    if not(input_par['Waveform_and_parametrisation']['spins'] in ['no-spins', 'aligned', 'precessing']): raise ValueError('Invalid spin option requested.')

    # ====================================#
    # Read training range and test point. #
    # ====================================#

    params_ranges = {}
    print('[Training_range]\n')
    for key in default_params_ranges:
        
        if((key=='ecc')      and not(input_par['Waveform_and_parametrisation']['eccentricity'])):        continue
        if(('lambda' in key) and not(input_par['Waveform_and_parametrisation']['tides'])):               continue
        if((key=='s1x')      and not(input_par['Waveform_and_parametrisation']['spins']=='precessing')): continue
        if((key=='s2x')      and not(input_par['Waveform_and_parametrisation']['spins']=='precessing')): continue
        if((key=='s1y')      and not(input_par['Waveform_and_parametrisation']['spins']=='precessing')): continue
        if((key=='s2y')      and not(input_par['Waveform_and_parametrisation']['spins']=='precessing')): continue
        if((key=='s1z')      and    (input_par['Waveform_and_parametrisation']['spins']=='no-spins'  )): continue
        if((key=='s2z')      and    (input_par['Waveform_and_parametrisation']['spins']=='no-spins'  )): continue

        keytype = type(default_params_ranges[key][0])
        try:
            params_ranges[key]=[keytype(Config.get('Training_range',key+'-min')), keytype(Config.get('Training_range',key+'-max'))]
            print("{name} : [{min},{max}]".format(          name=key.ljust(max_len_keyword), min=params_ranges[key][0], max=params_ranges[key][1]))
        except (KeyError, configparser.NoOptionError, TypeError):
            params_ranges[key]=default_params_ranges[key]
            print("{name} : [{min},{max}] (default)".format(name=key.ljust(max_len_keyword), min=params_ranges[key][0], max=params_ranges[key][1]))

        if(params_ranges[key][1] < params_ranges[key][0]): raise ValueError("{} upper bound is smaller than its lower bound.".format(key))

    print('\n')

    test_values = {}
    print('[Test_values]\n')
    for key in default_test_values:
        keytype = type(default_test_values[key])
        try:
            test_values[key]=keytype(Config.get('Test_values',key))
            print("{name} : {value}".format(          name=key.ljust(max_len_keyword), value=test_values[key]))
        except (KeyError, configparser.NoOptionError, TypeError):
            
            # Putting this block here allows to test the accuracy of the ROQ against regimes outside the training range (e.g. trained with tides=0 and checking the error with non-zero tides)
            if((key=='ecc')      and not(input_par['Waveform_and_parametrisation']['eccentricity'])):        continue
            if(('lambda' in key) and not(input_par['Waveform_and_parametrisation']['tides'])):               continue
            if((key=='s1x')      and not(input_par['Waveform_and_parametrisation']['spins']=='precessing')): continue
            if((key=='s2x')      and not(input_par['Waveform_and_parametrisation']['spins']=='precessing')): continue
            if((key=='s1y')      and not(input_par['Waveform_and_parametrisation']['spins']=='precessing')): continue
            if((key=='s2y')      and not(input_par['Waveform_and_parametrisation']['spins']=='precessing')): continue
            if((key=='s1z')      and    (input_par['Waveform_and_parametrisation']['spins']=='no-spins'  )): continue
            if((key=='s2z')      and    (input_par['Waveform_and_parametrisation']['spins']=='no-spins'  )): continue
            
            test_values[key]=default_test_values[key]
            print("{name} : {value} (default)".format(name=key.ljust(max_len_keyword), value=test_values[key]))
        if key in params_ranges.keys():
            if not(params_ranges[key][0] <= test_values[key] <= params_ranges[key][1]):
                warnings.warn("Chosen test value outside training range.")
    print('\n')

    return input_par, params_ranges, test_values

class PyROQ:
    """
    PyROQ Class
    
    * Works with a list of very basic waveform wrappers provided in wvfwrappers.py
    
    * The parameter space is *defined* by the keywords of 'params_ranges' 

    """

    def __init__(self,
                 config_pars,
                 params_ranges              = default_params_ranges,
                 start_values               = default_start_values ,
                 distance                   = 10                   , # [Mpc]. Dummy value, distance does not enter the interpolants construction
                 additional_waveform_params = {}                   , # Dictionary with any parameter needed for the waveform approximant
                 ):

        self.distance                   = distance
        self.additional_waveform_params = additional_waveform_params
        self.params_ranges              = params_ranges
        self.start_values               = start_values
        
        # Read input params
        self.approximant         = config_pars['Waveform_and_parametrisation']['approximant']

        self.mc_q_par            = config_pars['Waveform_and_parametrisation']['mc-q-par']
        self.spin_sph            = config_pars['Waveform_and_parametrisation']['spin-sph']

        self.f_min               = config_pars['Waveform_and_parametrisation']['f-min']
        self.f_max               = config_pars['Waveform_and_parametrisation']['f-max']
        self.deltaF              = 1./config_pars['Waveform_and_parametrisation']['seglen']
        
        self.n_basis_search_iter = config_pars['ROQ']['n-basis-search-iter']

        self.n_basis_low_lin     = config_pars['ROQ']['n-basis-low-lin']
        self.n_basis_hig_lin     = config_pars['ROQ']['n-basis-hig-lin']
        self.n_basis_step_lin    = config_pars['ROQ']['n-basis-step-lin']
        self.tolerance           = config_pars['ROQ']['tolerance']

        self.n_basis_low_quad    = config_pars['ROQ']['n-basis-low-quad']
        self.n_basis_hig_quad    = config_pars['ROQ']['n-basis-hig-quad']
        self.n_basis_step_quad   = config_pars['ROQ']['n-basis-step-quad']
        self.tolerance_quad      = config_pars['ROQ']['tolerance-quad']

        self.n_tests_basis       = config_pars['ROQ']['n-tests-basis']
        self.n_tests_post        = config_pars['ROQ']['n-tests-post']
        self.error_version       = config_pars['ROQ']['error-version']

        self.parallel            = config_pars['Parallel']['parallel']
        self.n_processes         = config_pars['Parallel']['n-processes']
        
        self.outputdir           = config_pars['I/O']['output']
        self.verbose             = config_pars['I/O']['verbose']
        
        # Sanity checks
        if(self.mc_q_par and (('m1' in self.params_ranges) or ('m2' in self.params_ranges))):
            raise ValueError("Cannot pass 'm1' or 'm2' in params_ranges with the 'mc_q_par' option activated.")
        elif(not(self.mc_q_par) and (not('m1' in self.params_ranges) or not('m2' in self.params_ranges))):
            raise ValueError("Need to pass 'm1' and 'm2' in params_ranges with the 'mc_q_par' option de-activated.")

        if(self.spin_sph and (('s1x' in self.params_ranges) or ('s1y' in self.params_ranges) or ('s1z' in self.params_ranges) or ('s2x' in self.params_ranges) or ('s2y' in self.params_ranges) or ('s2z' in self.params_ranges))):
            raise ValueError("Cannot pass 's1[xyz]' or 's2[xyz]' in params_ranges with the 'spin_sph' option activated.")
        if(not(self.spin_sph) and (('s1s1' in self.params_ranges) or ('s1s2' in self.params_ranges) or ('s1s3' in self.params_ranges) or ('s2s1' in self.params_ranges) or ('s2s2' in self.params_ranges) or ('s2s3' in self.params_ranges))):
            raise ValueError("Cannot pass 's1s[123]' or 's2s[123]' in params_ranges with the 'spin_sph' option de-activated.")

        if(not(self.n_basis_low_lin>1) or not(self.n_basis_low_quad>1)): raise ValueError("The minimum number of basis elements has to be larger than 1.")
        
        if not os.path.exists(self.outputdir):
            os.makedirs(self.outputdir)
            os.makedirs(os.path.join(self.outputdir, 'Plots'))
            os.makedirs(os.path.join(self.outputdir, 'ROQ_data'))
            os.makedirs(os.path.join(self.outputdir, 'ROQ_data/Linear'))
            os.makedirs(os.path.join(self.outputdir, 'ROQ_data/Quadratic'))
    
        # Choose waveform
        if self.approximant in WfWrapper.keys():
            self.wvf = WfWrapper[self.approximant](self.approximant, self.additional_waveform_params)
        else:
            raise ValueError('Unknown approximant requested.')

        # Build the map between params names and indexes
        self.map_params_indexs() # self.i2n, self.n2i, self.nparams
        
        # Initial basis
        self.freq = np.arange(self.f_min, self.f_max, self.deltaF)
        self.initial_basis() # self.params_low, self.params_hig, self.params_ini, self.hp1
        
    def howmany_within_range(self, row, minimum, maximum):
        """
        Returns how many numbers lie within `maximum` and `minimum` in a given `row`
        """
        count = 0
        for n in row:
            if minimum <= n <= maximum:
                count = count + 1
        return count

    def proj(self, u, v):
        """
        Calculating the projection of complex vector v on complex vector u
        Note: this algorithm assume denominator isn't zero
        """
        return u * np.vdot(v,u) / np.vdot(u,u) 

    def gram_schmidt(self, bases, vec):
        """
        Calculating the normalized residual (= a new basis term) of a vector vec from known bases
        """
        for i in np.arange(0,len(bases)):
            vec = vec - self.proj(bases[i], vec)
        return vec/np.sqrt(np.vdot(vec,vec)) # normalized new basis
    
    def overlap_of_two_waveforms(self, wf1, wf2):
        """
            Calculating overlap (FIXME: change to a more representative name) of two waveforms
        """
        
        # From the forked master version of the public PyROQ: https://github.com/qihongcat/PyROQ/blob/cb6350751dcff303957ace5ac83e6ff6e265a9c7/Code/PyROQ/pyroq.py#L40
        if(self.error_version=='v1'):
            wf1norm = wf1/np.sqrt(np.vdot(wf1,wf1))
            wf2norm = wf2/np.sqrt(np.vdot(wf2,wf2))
            measure = (1-np.real(np.vdot(wf1norm, wf2norm)))*self.deltaF
        # From the PyROQ paper: https://arxiv.org/abs/2009.13812
        elif(self.error_version=='v2'):
            diff    = wf1 - wf2
            measure = np.real(np.vdot(diff, diff))*self.deltaF
        # From the forked master version of the public PyROQ (commented): https://github.com/qihongcat/PyROQ/blob/cb6350751dcff303957ace5ac83e6ff6e265a9c7/Code/PyROQ/pyroq.py#L39
        elif(self.error_version=='v3'):
            diff    = wf1 - wf2
            measure = 1 - 0.5*np.real(np.vdot(diff, diff))
        # Same as 'v3', but without the (1-0.5*) factor
        elif(self.error_version=='v4'):
            diff    = wf1 - wf2
            measure = np.real(np.vdot(diff, diff))
        else:
            raise VersionError
        
        return measure

    def spherical_to_cartesian(self, sph):
        x = sph[0]*np.sin(sph[1])*np.cos(sph[2])
        y = sph[0]*np.sin(sph[1])*np.sin(sph[2])
        z = sph[0]*np.cos(sph[1])
        return [x,y,z]

    def get_m1m2_from_mcq(self, mc,q):
        m2 = mc * q ** (-0.6) * (1+q)**0.2
        m1 = m2 * q
        return np.array([m1,m2])

    def mass_range(self,mc_low, mc_high, q_low, q_high):
        mmin = self.get_m1m2_from_mcq(mc_low,q_high)[1]
        mmax = self.get_m1m2_from_mcq(mc_high,q_high)[0]
        return [mmin, mmax]

    def map_params_indexs(self):
        """
        Build a map between the parameters names and the indexes of
        the parameter arrays, and its inverse
        """
        names = self.params_ranges.keys()
        self.nparams = len(names)
        self.n2i = dict(zip(names,range(self.nparams)))
        self.i2n = {i: n for n, i in self.n2i.items()}
        return

    def update_waveform_params(self, paramspoint):
        """
        Update the waveform parameters (dictionary) with those in
        paramspoint (np array)
        """
        p = {}
        for i,k in self.i2n.items():
            p[k] = paramspoint[i]

        return p
         
    def generate_params_points(self,npts=0,round_to_digits=6):
        """
        Uniformly sample the parameter arrays
        """
        if npts <= 0: npts = self.n_basis_search_iter
        paramspoints = np.random.uniform(self.params_low,
                                         self.params_hig,
                                         size=(npts,self.nparams))
        return paramspoints.round(decimals=round_to_digits)
    
    def _paramspoint_to_wave(self, paramspoint):
        """
        Generate a waveform given a paramspoint
        By default, if paramspoint contains the spherical spin, then updates the cartesian accordingly.
        """
        p = self.update_waveform_params(paramspoint)

        if self.mc_q_par: p['m1'],p['m2'] = self.get_m1m2_from_mcq(p['mc'],p['q'])

        if self.spin_sph:
            s1sphere_tmp               = p['s1s1'],p['s1s2'],p['s1s3']
            p['s1x'],p['s1y'],p['s1z'] = self.spherical_to_cartesian(s1sphere_tmp)
            s2sphere_tmp               = p['s2s1'],p['s2s2'],p['s2s3']
            p['s2x'],p['s2y'],p['s2z'] = self.spherical_to_cartesian(s2sphere_tmp)
    
        # We build a linear basis only for hp, since empirically the same basis accurately works to represent hc too (see [arXiv:1604.08253).
        hp, hc = self.wvf.generate_waveform(p, self.deltaF, self.f_min, self.f_max, self.distance)
        return hp, hc

    def _compute_modulus(self, paramspoint, known_bases, term):

        hp, _ = self._paramspoint_to_wave(paramspoint)

        if   term == 'lin' : residual = hp
        elif term == 'quad': residual = (np.absolute(hp))**2
        else               : raise TermError

        h_to_proj = residual

        for k in np.arange(0,len(known_bases)):
            residual -= self.proj(known_bases[k],h_to_proj)
        
        return np.sqrt(np.vdot(residual, residual))
        
    def _least_match_waveform_unnormalized(self, paramspoints, known_bases, term):

        # Generate npts random waveforms.
        if self.parallel:
            paramspointslist = paramspoints.tolist()
            pool = mp.Pool(processes=n_processes)
            modula = [pool.apply(self._compute_modulus, args=(paramspoint, known_bases, term)) for paramspoint in paramspointslist]
            pool.close()
        else:
            npts   = len(paramspoints) # = self.npts
            modula = np.zeros(npts)
            for i,paramspoint in enumerate(paramspoints):
                modula[i] = np.real(self._compute_modulus(paramspoint, known_bases, term))

        # Select the worst represented waveform (in terms of the previous known basis)
        arg_newbasis = np.argmax(modula) 
        hp, _ = self._paramspoint_to_wave(paramspoints[arg_newbasis])
        if   term == 'lin' : pass
        elif term == 'quad': hp = (np.absolute(hp))**2
        else               : raise TermError
        
        # Extract the linearly independent part of the worst represented waveform, which constitutes a new basis element.
        # Note: the new basis element is not a 'waveform', since subtraction of two waveforms does not generate a waveform.
        basis_new = self.gram_schmidt(known_bases, hp)
       
        return np.array([basis_new, paramspoints[arg_newbasis], modula[arg_newbasis]]) # elements, masses&spins&lambdas, residual mod
            
    def _bases_searching_results_unnormalized(self, known_bases, basis_waveforms, params, residual_modula, term):
        
        if term == 'lin':
            n_basis_hig = self.n_basis_hig_lin
            file_bases  = self.outputdir+'/ROQ_data/Linear/linear_bases.npy'
            file_params = self.outputdir+'/ROQ_data/Linear/linear_bases_waveform_params.npy'
        elif term=='quad':
            n_basis_hig = self.n_basis_hig_quad
            file_bases  = self.outputdir+'/ROQ_data/Quadratic/quadratic_bases.npy'
            file_params = self.outputdir+'/ROQ_data/Quadratic/quadratic_basis_waveform_params.npy'
        else:
            raise TermError
    
        # This block generates a basis of dimension nbases (maximum dimension selected by the user).
        print('\n\n###########################\n# Starting {} iteration #\n###########################\n'.format(term.ljust(4)))
        for k in np.arange(0,n_basis_hig-1):
            
            # Generate npts random waveforms.
            paramspoints = self.generate_params_points()
            
            # From the npts randomly generated waveforms, select the worst represented one (i.e. with the largest residuals after basis projection).
            basis_new, params_new, rm_new = self._least_match_waveform_unnormalized(paramspoints, known_bases, term)
            if self.verbose:
                np.set_printoptions(suppress=True)
                print("Iter: ".format(term), k+1, " -- New basis waveform:", params_new)
                np.set_printoptions(suppress=False)

            # The worst represented waveform becomes the new basis element.
            known_bases     = np.append(known_bases,     np.array([basis_new]),  axis=0)
            params          = np.append(params,          np.array([params_new]), axis=0)
            residual_modula = np.append(residual_modula, rm_new)

        # Store the constructed largest basis. If its dimension is enough to stay below tolerance, the ROQ greedy algorithm will downselect this to a smaller number, the minumum required to stay below tolerance.
        np.save(file_bases,  known_bases)
        np.save(file_params, params     )

        return known_bases, params, residual_modula
    
    def initial_basis(self):
        """
        Initialize parameter ranges and basis.
        """
        if self.verbose:
            print('\n\n######################\n# Initialising basis #\n######################\n')
            print('nparams = {}\n'.format(self.nparams))
            print('index | name    | ( min - max )           | start')

        self.params_low, self.params_hig, params_ini_list = [], [], []
        # Set bounds
        for i,n in self.i2n.items():
            self.params_low.append(self.params_ranges[n][0])
            self.params_hig.append(self.params_ranges[n][1])
            params_ini_list.append(self.start_values[n])

            if self.verbose:
                print('{}    | {} | ( {:.6f} - {:.6f} ) | {:.6f}'.format(str(i).ljust(2),
                                                                      n.ljust(len('lambda1')),
                                                                      self.params_low[i],
                                                                      self.params_hig[i],
                                                                      params_ini_list[i]))
        self.params_ini = np.array([params_ini_list])
        # First waveform
        self.hp1, _ = self._paramspoint_to_wave(params_ini_list)
        
        return 

    def empirical_nodes(self, ndim, known_bases, fact=100000000):
        
        """
        Generate the empirical interpolation nodes from a given basis.
        Follows the algorithm detailed in Ref. Phys. Rev. X 4, 031006, according to PRD 104, 063031 (2021).
        """
        
        emp_nodes    = np.arange(0,ndim) * fact
        emp_nodes[0] = np.argmax(np.absolute(known_bases[0]))
        
        c1           = known_bases[1,emp_nodes[0]]/known_bases[0,1]
        interp1      = np.multiply(c1,known_bases[0])
        diff1        = interp1 - known_bases[1]
        r1           = np.absolute(diff1)
        emp_nodes[1] = np.argmax(r1)
        
        for k in np.arange(2,ndim):
            
            emp_tmp      = emp_nodes[0:k]
            Vtmp         = np.transpose(known_bases[0:k,emp_tmp])
            inverse_Vtmp = np.linalg.pinv(Vtmp)
            e_to_interp  = known_bases[k]
            Ci           = np.dot(inverse_Vtmp, e_to_interp[emp_tmp])
            interpolantA = np.zeros(len(known_bases[k]))+np.zeros(len(known_bases[k]))*1j
            
            for j in np.arange(0, k):
                tmp           = np.multiply(Ci[j], known_bases[j])
                interpolantA += tmp
            
            diff         = interpolantA - known_bases[k]
            r            = np.absolute(diff)
            emp_nodes[k] = np.argmax(r)
            emp_nodes    = sorted(emp_nodes)
    
        u, c      = np.unique(emp_nodes, return_counts=True)
        dup       = u[c > 1]
        emp_nodes = np.unique(emp_nodes)
        ndim      = len(emp_nodes)
        V         = np.transpose(known_bases[0:ndim, emp_nodes])
        inverse_V = np.linalg.pinv(V)
        
        return np.array([ndim, inverse_V, emp_nodes])

    def _roq_error_from_basis(self, ndim, inverse_V, emp_nodes, known_bases, paramspoint, term):
        
        # Create benchmark waveform
        hp, _ = self._paramspoint_to_wave(paramspoint)
        if   term == 'lin' : pass
        elif term == 'quad': hp = (np.absolute(hp))**2
        else               : raise TermError
        
        # Initialise the interpolant
        interpolantA = np.zeros(len(hp))+np.zeros(len(hp))*1j
        
        # Compute the coefficients c_i of the interpolant written in terms of the basis.
        Ci           = np.dot(inverse_V, hp[emp_nodes])
        
        # Construct the interpolant, summing over each basis element.
        for j in np.arange(0, ndim):
            tmp           = np.multiply(Ci[j], known_bases[j])
            interpolantA += tmp
        
        # Return the goodness-of-interpolation measure
        return self.overlap_of_two_waveforms(hp, interpolantA)
    
    def _roq_error_check(self, ndim, inverse_V, emp_nodes, known_bases, term):
        
        """
            Basis construction stopping function.
            Compute the overlap representation error on a set of random points, using the given basis. If all of them are below tolerance, this is the basis we are searching for.
        """
        
        # Initialise tolerance, parameter space and data structures.
        if   term == 'lin':  tol = self.tolerance
        elif term == 'quad': tol = self.tolerance_quad
        else:                raise TermError
        paramspoints = self.generate_params_points(npts=self.n_tests_basis)
        surros       = np.zeros(self.n_tests_basis)
        count        = 0
        
        # Compute the overlap representation error
        for i, paramspoint in enumerate(paramspoints):
            surros[i] = self._roq_error_from_basis(ndim,
                                                   inverse_V,
                                                   emp_nodes,
                                                   known_bases[0:ndim],
                                                   paramspoint,
                                                   term)
            # Store outliers
            if (surros[i] > tol):
                print("Found outlier: ", paramspoint, " with surrogate error ", surros[i])
                count = count+1
    
        if self.verbose:
            print('\n{}'.format(ndim), "basis elements gave", count, "bad points of surrogate error >", self.tolerance, '\n')
        if count == 0: return 0
        else:          return 1
    
    def _roqs(self, known_bases, term):
        
        # Initialise iteration and create paths in which to store the output.
        if term == 'lin':
            n_basis_low  = self.n_basis_low_lin
            n_basis_hig  = self.n_basis_hig_lin
            n_basis_step = self.n_basis_step_lin
            froq         = self.outputdir+'/ROQ_data/Linear/B_linear.npy'
            fnodes       = self.outputdir+'/ROQ_data/Linear/fnodes_linear.npy'
        elif term == 'quad':
            n_basis_low  = self.n_basis_low_quad
            n_basis_hig  = self.n_basis_hig_quad
            n_basis_step = self.n_basis_step_quad
            froq         = self.outputdir+'/ROQ_data/Quadratic/B_quadratic.npy'
            fnodes       = self.outputdir+'/ROQ_data/Quadratic/fnodes_quadratic.npy'
        else:
              raise TermError

        # Start from a user-selected minimum number of basis elements and keep adding elements until that basis represents well enough a sufficient number of random waveforms or until you hit the user-selected maximum basis elements number.
        flag = 0
        for num in np.arange(n_basis_low, n_basis_hig+1, n_basis_step):
            
            # Build the empirical interpolation nodes for this basis.
            ndim, inverse_V, emp_nodes = self.empirical_nodes(num, known_bases)
            
            # If the overlap representation error is below tolerance, stop the iteration and store this basis.
            if(self._roq_error_check(ndim, inverse_V, emp_nodes, known_bases, term) == 0):
                
                # Build the interpolant from the given basis.
                b = np.dot(np.transpose(known_bases[0:ndim]),inverse_V)
                f = self.freq[emp_nodes]
                
                # Store the output.
                np.save(froq,np.transpose(b))
                np.save(fnodes,f)
                
                if self.verbose:
                    print("Number of {} basis elements is".format(term), ndim, "and the ROQ data are saved in",froq, '\n')
                flag = 1
                break

        if not(flag): raise Exception('Could not find a basis to correctly represent the model within the given tolerance and maximum dimension selected.\nTry increasing the allowed basis size or decreasing the tolerance.')

        return b,f

    ## Main function starting the ROQ construction.

    def run(self):
        
        # Initialise data.
        d          = {}
        
        # Initialise basis.
        hp1        = self.hp1
        hp1_quad   = (np.absolute(hp1))**2
        params_ini = self.params_ini

        known_bases_start     = np.array([hp1/np.sqrt(np.vdot(hp1,hp1))])
        basis_waveforms_start = np.array([hp1])
        residual_modula_start = np.array([0.0])
        
        # Construct the linear basis.
        # Also store the basis in a file, so that it can be re-used in the next iterations, if the currently selected maximum number of basis elements is too small to meet the required tolerance.
        bases, params, residual_modula = self._bases_searching_results_unnormalized(known_bases_start,
                                                                                    basis_waveforms_start,
                                                                                    params_ini,
                                                                                    residual_modula_start,
                                                                                    'lin')

        # From the linear basis constructed above, extract:
        # i) the empirical interpolation nodes (i.e. the subset of frequencies on which the ROQ rule is evaluated);
        # ii) the basis interpolant, which allows to construct an arbitrary waveform at an arbitrary frequency point from the constructed basis.
        B, f = self._roqs(bases, 'lin')
            
        # Internally store the output data for later testing.
        d['lin_B']          = B
        d['lin_f']          = f
        d['lin_bases']      = bases
        d['lin_params']     = params
        d['lin_res']        = residual_modula
        d['lin_emp_nodes']  = np.searchsorted(self.freq, d['lin_f'])


        # Repeat the same as above for the quadratic terms.
        # FIXME: Should be inserted in a loop and not repeated.
        known_bases_start     = np.array([hp1_quad/np.sqrt(np.vdot(hp1_quad,hp1_quad))])
        basis_waveforms_start = np.array([hp1_quad])
        residual_modula_start = np.array([0.0])
        
        bases, params, residual_modula = self._bases_searching_results_unnormalized(known_bases_start,
                                                                                    basis_waveforms_start,
                                                                                    params_ini,
                                                                                    residual_modula_start,
                                                                                    'quad')
        B, f = self._roqs(bases, 'quad')
        
        d['quad_B']         = B
        d['quad_f']         = f
        d['quad_bases']     = bases
        d['quad_params']    = params
        d['quad_res']       = residual_modula
        d['quad_emp_nodes'] = np.searchsorted(self.freq, d['quad_f'])
        
        return d
    
    ## Functions to test the performance of the waveform representation, using the interpolant built from the selected basis.
    
    def plot_representation_error(self, b, emp_nodes, paramspoint, term):
        
        hp, hc = self._paramspoint_to_wave(paramspoint)
        
        if   term == 'lin' :
            pass
        elif term == 'quad':
            hphc = np.real(hp * np.conj(hc))
            hp   = (np.absolute(hp))**2
            hc   = (np.absolute(hc))**2
        else               :
            raise TermError
        
        freq           = self.freq
        hp_emp, hc_emp = hp[emp_nodes], hc[emp_nodes]
        hp_rep, hc_rep = np.dot(b,hp_emp), np.dot(b,hc_emp)
        diff_hp        = hp_rep - hp
        diff_hc        = hc_rep - hc
        rep_error_hp   = diff_hp/np.sqrt(np.vdot(hp,hp))
        rep_error_hc   = diff_hc/np.sqrt(np.vdot(hc,hc))
        if term == 'quad':
            hphc_emp       = hphc[emp_nodes]
            hphc_rep       = np.dot(b,hphc_emp)
            diff_hphc      = hphc_rep - hphc
            rep_error_hphc = diff_hphc/np.sqrt(np.vdot(hphc,hphc))
            
        plt.figure(figsize=(8,5))
        plt.plot(freq, np.real(hp),     label='Real part of h_+ (full)')
        plt.plot(freq, np.real(hp_rep), label='Real part of h_+ (ROQ)')
        plt.xlabel('Frequency')
        plt.ylabel('Waveform')
        plt.title('Waveform comparison ({})'.format(term.ljust(4)))
        plt.legend(loc=0)
        plt.savefig(os.path.join(self.outputdir,'Plots/Waveform_comparison_hp_real_{}.png'.format(term)))

        plt.figure(figsize=(8,5))
        plt.plot(freq, np.imag(hp),     label='Imag part of h_+ (full)')
        plt.plot(freq, np.imag(hp_rep), label='Imag part of h_+ (ROQ)')
        plt.xlabel('Frequency')
        plt.ylabel('Waveform')
        plt.title('Waveform comparison ({})'.format(term.ljust(4)))
        plt.legend(loc=0)
        plt.savefig(os.path.join(self.outputdir,'Plots/Waveform_comparison_hp_imag_{}.png'.format(term)))

        plt.figure(figsize=(8,5))
        plt.plot(freq, np.real(hc),     label='Real part of h_x (full)')
        plt.plot(freq, np.real(hc_rep), label='Real part of h_x (ROQ)')
        plt.xlabel('Frequency')
        plt.ylabel('Waveform')
        plt.title('Waveform comparison ({})'.format(term.ljust(4)))
        plt.legend(loc=0)
        plt.savefig(os.path.join(self.outputdir,'Plots/Waveform_comparison_hc_real_{}.png'.format(term)))

        plt.figure(figsize=(8,5))
        plt.plot(freq, np.imag(hc),     label='Imag part of h_x (full)')
        plt.plot(freq, np.imag(hc_rep), label='Imag part of h_x (ROQ)')
        plt.xlabel('Frequency')
        plt.ylabel('Waveform')
        plt.title('Waveform comparison ({})'.format(term.ljust(4)))
        plt.legend(loc=0)
        plt.savefig(os.path.join(self.outputdir,'Plots/Waveform_comparison_hc_imag_{}.png'.format(term)))

        if term == 'quad':
            plt.figure(figsize=(8,5))
            plt.plot(freq, hphc,     label='Real part of h_+ * conj(h_x) (full)')
            plt.plot(freq, hphc_rep, label='Real part of h_+ * conj(h_x) (ROQ)')
            plt.xlabel('Frequency')
            plt.ylabel('Waveform')
            plt.title('Waveform comparison ({})'.format(term.ljust(4)))
            plt.legend(loc=0)
            plt.savefig(os.path.join(self.outputdir,'Plots/Waveform_comparison_hphc_real_{}.png'.format(term)))

            plt.figure(figsize=(8,5))
            plt.plot(freq, rep_error_hphc, label='Real part of (h_+ * conj(h_x))')
            plt.xlabel('Frequency')
            plt.ylabel('Fractional Representation Error')
            plt.title('Representation Error ({})'.format(term.ljust(4)))
            plt.legend(loc=0)
            plt.savefig(os.path.join(self.outputdir,'Plots/Representation_error_hp_{}.png'.format(term)))

        plt.figure(figsize=(8,5))
        plt.plot(freq, np.real(rep_error_hp), label='Real      part of h_+')
        plt.plot(freq, np.imag(rep_error_hp), label='Imaginary part of h_+')
        plt.xlabel('Frequency')
        plt.ylabel('Fractional Representation Error')
        plt.title('Representation Error ({})'.format(term.ljust(4)))
        plt.legend(loc=0)
        plt.savefig(os.path.join(self.outputdir,'Plots/Representation_error_hp_{}.png'.format(term)))

        plt.figure(figsize=(8,5))
        plt.plot(freq, np.real(rep_error_hc), label='Real      part of h_x')
        plt.plot(freq, np.imag(rep_error_hc), label='Imaginary part of h_x')
        plt.xlabel('Frequency')
        plt.ylabel('Fractional Representation Error')
        plt.title('Representation Error ({})'.format(term.ljust(4)))
        plt.legend(loc=0)
        plt.savefig(os.path.join(self.outputdir,'Plots/Representation_error_hc_{}.png'.format(term)))

        return
    
    def test_roq_error(self, b, emp_nodes, term):
        
        # Initialise structures
        nsamples  = self.n_tests_post
        ndim      = len(emp_nodes)
        surros_hp = np.zeros(nsamples)
        surros_hc = np.zeros(nsamples)
        
        # Draw random test points
        paramspoints = self.generate_params_points(npts=nsamples)
        
        # Select tolerance
        if   term == 'lin':
            tol = self.tolerance
        elif term == 'quad':
            tol = self.tolerance_quad
            surros_hphc = np.zeros(nsamples)
        else:
            raise TermError
        
        # Start looping over test points
        print('\n\n###########################################\n# Starting surrogate tests {} iteration #\n###########################################\n'.format(term.ljust(4)))
        print('Tolerance: ', tol)
        for i,paramspoint in enumerate(paramspoints):
            
            # Generate test waveform
            hp, hc = self._paramspoint_to_wave(paramspoint)
            
            # Compute quadratic terms and interpolant representations
            if term == 'quad':
                hphc     = np.real(hp * np.conj(hc))
                hphc_emp = hphc[emp_nodes]
                hphc_rep = np.dot(b,hphc_emp)
            
                hp       = (np.absolute(hp))**2
                hc       = (np.absolute(hc))**2

            hp_emp    = hp[emp_nodes]
            hp_rep    = np.dot(b,hp_emp)
            hc_emp    = hc[emp_nodes]
            hc_rep    = np.dot(b,hc_emp)

            # Compute the representation error. This is the same measure employed to stop adding elements to the basis
            surros_hp[i] = self.overlap_of_two_waveforms(hp, hp_rep)
            surros_hc[i] = self.overlap_of_two_waveforms(hc, hc_rep)
            if term == 'quad':
                surros_hphc[i] = self.overlap_of_two_waveforms(hphc, hphc_rep)

            # If a test case exceeds the error, let the user know. Always print typical test result every 100 steps
            np.set_printoptions(suppress=True)
            if self.verbose:
                if (surros_hp[i] > tol): print("h_+     above tolerance: Iter: ", i, "Surrogate value: ", surros_hp[i], "Parameters: ", paramspoints[i])
                if (surros_hc[i] > tol): print("h_x     above tolerance: Iter: ", i, "Surrogate value: ", surros_hc[i], "Parameters: ", paramspoints[i])
#                if ((term == 'quad') and (surros_hphc[i] > tol)):
#                    print("h_+ h_x above tolerance: Iter: ", i, "Surrogate value: ", surros_hphc[i], "Parameters: ", paramspoints[i])
                if i%100==0:
                    print("h_+     rolling check (every 100 steps): Iter: ",             i, "Surrogate value: ", surros_hp[i])
                    print("h_x     rolling check (every 100 steps): Iter: ",             i, "Surrogate value: ", surros_hc[i])
#                    if (term == 'quad'):
#                        print("h_+ h_x rolling check (every 100 steps): Iter: ",             i, "Surrogate value: ", surros_hphc[i])
            np.set_printoptions(suppress=False)
    
        # Plot the test results
        plt.figure(figsize=(8,5))
        plt.semilogy(surros_hp,'o', label='h_+')
        plt.semilogy(surros_hc,'o', label='h_x')
#        if term == 'quad':
#            plt.semilogy(surros_hphc,'o', label='h_+ * conj(h_x)')
        plt.xlabel("Number of Random Test Points")
        plt.ylabel("Surrogate Error ({})".format(term.ljust(4)))
        plt.legend(loc=0)
        plt.savefig(os.path.join(self.outputdir,"Plots/Surrogate_errors_random_test_points_{}.png".format(term)))
    
        return

if __name__ == '__main__':

    parser         = OptionParser(usage)
    parser.add_option('--config-file', type='string', metavar = 'config_file', default = None)

    (opts,args) = parser.parse_args()
    config_file = opts.config_file

    # Read ROQ parameters, and load training ranges and a point of the parameter space on which a targeted check is required.
    config_pars, params_ranges, test_values = read_config(config_file)
    
    # Convert to LAL identification number, if passing a LAL approximant
    try:    config_pars['Waveform_and_parametrisation']['approximant'] = lalsimulation.SimInspiralGetApproximantFromString(config_pars['Waveform_and_parametrisation']['approximant'])
    except: pass

    # approx = 'teobresums-giotto'
    # params_ranges = {
    #     'mc'      : [30, 31]    ,
    #     'q'       : [1, 1.2]    ,
    #     's1x'     : [0, 0]      ,
    #     's1y'     : [0, 0]      ,
    #     's1z'     : [-0.5, 0.5] ,
    #     's2x'     : [0, 0]      ,
    #     's2y'     : [0, 0]      ,
    #     's2z'     : [-0.5, 0.5] ,
    #     'lambda1' : [0, 1000]   ,
    #     'lambda2' : [0, 1000]   ,
    #     'iota'    : [0, np.pi]  ,
    #     'phiref'  : [0, 2*np.pi],
    # }

    #approx = 'mlgw-bns'
    #    params_ranges = {
    #        'mc'      : [0.9, 0.92] ,
    #        'q'       : [1, 1.02]   ,
    #        's1x'     : [0, 0]      ,
    #        's1y'     : [0, 0]      ,
    #        's1z'     : [-0.5, 0.5] ,
    #        's2x'     : [0, 0]      ,
    #        's2y'     : [0, 0]      ,
    #        's2z'     : [-0.5, 0.5] ,
    #        'lambda1' : [5, 50]     ,
    #        'lambda2' : [5, 50]     ,
    #        'iota'    : [0, np.pi]  ,
    #        'phiref'  : [0, 2*np.pi],
    #    }

    start_values = {'{}'.format(key): params_ranges[key][0]  for key in params_ranges.keys()}

    # Point(s) of the parameter space on which to initialise the basis. If not passed by the user, select defaults.
    if not('start_values'  in locals() or 'start_values'  in globals()): start_values  = default_start_values

    # Initialise ROQ
    pyroq = PyROQ(
                  config_pars,
                  params_ranges = params_ranges,
                  start_values  = start_values,
                  )

    ##############################################
    # No parameter should be changed below here. #
    ##############################################

    freq = pyroq.freq

    if not(config_pars['I/O']['post-processing-only']):
        # Create the bases and save them
        data                   = pyroq.run()
    else:
        data                   = {}
        data['lin_f']          = np.load(output+'/ROQ_data/Linear/fnodes_linear.npy')
        data['lin_B']          = np.load(output+'/ROQ_data/Linear/B_linear.npy')
        data['quad_f']         = np.load(output+'/ROQ_data/Quadratic/fnodes_quadratic.npy')
        data['quad_B']         = np.load(output+'/ROQ_data/Quadratic/B_quadratic.npy')
        data['lin_emp_nodes']  = np.searchsorted(freq, data['lin_f'])
        data['quad_emp_nodes'] = np.searchsorted(freq, data['quad_f'])

        print('check np.traspose in storing B output.')
        raise Exception("Not yet completed.")

    print('\n###########\n# Results #\n###########\n')
    print('Linear    basis reduction factor: (Original freqs [{}]) / (New freqs [{}]) = {}'.format(len(freq), len(data['lin_f']),  len(freq)/len(data['lin_f'])))
    print('Quadratic basis reduction factor: (Original freqs [{}]) / (New freqs [{}]) = {}'.format(len(freq), len(data['quad_f']), len(freq)/len(data['quad_f'])))

    # Test waveform
    print('\n\n#############################################\n# Testing the waveform using the parameters:#\n#############################################\n')
    parampoint = []
    print('name    | value | index')
    for name, val in test_values.items():
        print('{} | {}   | {} '.format(name.ljust(len('lambda1')), val, pyroq.n2i[name]))
        parampoint.append(val)
    parampoint = np.array(parampoint)

    # Surrogate tests
    pyroq.test_roq_error(data['lin_B'] , data['lin_emp_nodes'] , 'lin')
    pyroq.test_roq_error(data['quad_B'], data['quad_emp_nodes'], 'quad')

    # Now plot the representation error for a random waveform, using the interpolant built from the constructed basis. Useful for visual diagnostics.
    pyroq.plot_representation_error(data['lin_B'] , data['lin_emp_nodes'] , parampoint, 'lin')
    pyroq.plot_representation_error(data['quad_B'], data['quad_emp_nodes'], parampoint, 'quad')

    if(config_pars['I/O']['show-plots']): plt.show()
