import numpy as np, os, warnings
try:
    import configparser
except ImportError:
    import ConfigParser as configparser

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

# This is the training range of the 'mlgw-bns' approximant
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
