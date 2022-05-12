import numpy as np, os, warnings
try:
    import configparser
except ImportError:
    import ConfigParser as configparser

#Description of the package. Printed on stdout if --help option is give.
usage="""\n\n pyroq.py --config-file config.ini\n
Package description FIXME.

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

               approximant          Waveform approximant. Allowed values: ['teobresums-giotto', 'mlgw-bns', 'IMRPhenomPv2', 'IMRPhenomPv3', 'IMRPhenomXHM', 'TaylorF2Ecc', 'IMRPhenomPv2_NRTidal', 'IMRPhenomNSBH']. Default: 'teobresums-giotto'.
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
               tolerance-lin        Interpolation error threshold for linear basis elements. Default: 1e-8.
           
               n-basis-low-qua      Same as above, for quadratic basis. Default: 10.
               n-basis-hig-qua      Same as above, for quadratic basis. Usually 66% of linear basis one. Default: 20.
               n-basis-step-qua     Same as above, for quadratic basis. Default: 1.
               tolerance-qua        Same as above, for quadratic basis. Default: 1e-5.
               
       **************************************************************************
       * Parameters range and test values syntax.                               *
       **************************************************************************
       
       Allowed parameter names and units are:
       
               mc   (mc-q-par=1) : chirp mass
               q    (mc-q-par=1) : mass ratio
               m1   (mc-q-par=0) : mass object 1 [Mo]
               m2   (mc-q-par=0) : mass object 2 [Mo]
               s1s1 (spin-sph=1) : spin components object 1, spherical coords (FIXME: SPECIFY)
               s1s2 (spin-sph=1) : spin components object 1, spherical coords (FIXME: SPECIFY)
               s1s3 (spin-sph=1) : spin components object 1, spherical coords (FIXME: SPECIFY)
               s2s1 (spin-sph=1) : spin components object 2, spherical coords (FIXME: SPECIFY)
               s2s2 (spin-sph=1) : spin components object 2, spherical coords (FIXME: SPECIFY)
               s2s3 (spin-sph=1) : spin components object 2, spherical coords (FIXME: SPECIFY)
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

      FIXME description of distance [Mpc]

    """

# This is the training range of the 'mlgw-bns' approximant
default_params_ranges = {
    'mc'      : [0.9, 1.4]     ,
    'q'       : [1.0, 3.0]     ,
    'm1'      : [1.0, 3.0]     ,
    'm2'      : [0.5, 2.0]     ,
    's1x'     : [0.0, 0.0]     ,
    's1y'     : [0.0, 0.0]     ,
    's1z'     : [-0.5, 0.5]    ,
    's2x'     : [0.0, 0.0]     ,
    's2y'     : [0.0, 0.0]     ,
    's2z'     : [-0.5, 0.5]    ,
    's1s1'    : [0.0, 0.5]     , # mlgw-bns is non-precessing, but these values are to set conventions and it won't be called with spherical spin coords anyway (an error will be raised in such a case).
    's1s2'    : [0.0, np.pi]   ,
    's1s3'    : [0.0, 2*np.pi] ,
    's2s1'    : [0.0, 0.5]     ,
    's2s2'    : [0.0, np.pi]   ,
    's2s3'    : [0.0, 2*np.pi] ,
    'lambda1' : [5.0, 5000.0]  ,
    'lambda2' : [5.0, 5000.0]  ,
    'ecc'     : [0.0, 0.0]     ,
    'iota'    : [0.0, np.pi]   ,
    'phiref'  : [0.0, 2*np.pi] ,
}

default_test_values = {
        'mc'      : 1.3    ,
        'q'       : 2.0    ,
        'm1'      : 1.5    , 
        'm2'      : 1.5    ,
        's1x'     : 0.0    ,
        's1y'     : 0.0    ,
        's1z'     : 0.2    ,
        's2x'     : 0.0    ,
        's2y'     : 0.0    ,
        's2z'     : 0.1    ,
        's1s1'    : 0.3    ,
        's1s2'    : 0.4    ,
        's1s3'    : 0.5    ,
        's2s1'    : 0.3    ,
        's2s2'    : 0.4    ,
        's2s3'    : 0.5    ,
        'lambda1' : 1000.0 ,
        'lambda2' : 1000.0 ,
        'ecc'     : 0.0    ,
        'iota'    : 1.9    ,
        'phiref'  : 0.6    ,
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
                                                 'tolerance-lin'       : 1e-8,
                                               
                                                 'n-basis-low-qua'     : 20,
                                                 'n-basis-hig-qua'     : 80,
                                                 'n-basis-step-qua'    : 10,
                                                 'tolerance-qua'       : 1e-10,
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

    if(not(input_par['ROQ']['n-basis-low-lin']>1) or not(input_par['ROQ']['n-basis-low-qua']>1)): raise ValueError("The minimum number of basis elements has to be larger than 1.")
    if(input_par['Waveform_and_parametrisation']['spin-sph'] and not(input_par['Waveform_and_parametrisation']['spins']=='precessing')):
        raise ValueError('Spherical spin coordinates are currently supported only for precessing waveforms.')

    if not(input_par['Waveform_and_parametrisation']['spins'] in ['no-spins', 'aligned', 'precessing']): raise ValueError('Invalid spin option requested.')

    # Create dir structure
    if not os.path.exists(input_par['I/O']['output']):
        os.makedirs(input_par['I/O']['output'])
        os.makedirs(os.path.join(input_par['I/O']['output'], 'Plots'))
        os.makedirs(os.path.join(input_par['I/O']['output'], 'ROQ_data'))
        os.makedirs(os.path.join(input_par['I/O']['output'], 'ROQ_data/Linear'))
        os.makedirs(os.path.join(input_par['I/O']['output'], 'ROQ_data/Quadratic'))

    # ====================================#
    # Read training range and test point. #
    # ====================================#

    params_ranges = {}
    print('[Training_range]\n')
    for key in default_params_ranges:
        
        if((key=='m1')       and    (input_par['Waveform_and_parametrisation']['mc-q-par'])           ): continue
        if((key=='m2')       and    (input_par['Waveform_and_parametrisation']['mc-q-par'])           ): continue
        if((key=='mc')       and not(input_par['Waveform_and_parametrisation']['mc-q-par'])           ): continue
        if((key=='q')        and not(input_par['Waveform_and_parametrisation']['mc-q-par'])           ): continue
        if((key=='s1s1')     and not(input_par['Waveform_and_parametrisation']['spin-sph'])           ): continue
        if((key=='s1s2')     and not(input_par['Waveform_and_parametrisation']['spin-sph'])           ): continue
        if((key=='s1s3')     and not(input_par['Waveform_and_parametrisation']['spin-sph'])           ): continue
        if((key=='s2s1')     and not(input_par['Waveform_and_parametrisation']['spin-sph'])           ): continue
        if((key=='s2s2')     and not(input_par['Waveform_and_parametrisation']['spin-sph'])           ): continue
        if((key=='s2s3')     and not(input_par['Waveform_and_parametrisation']['spin-sph'])           ): continue
        if((key=='s1x')      and    (input_par['Waveform_and_parametrisation']['spin-sph'])           ): continue
        if((key=='s1y')      and    (input_par['Waveform_and_parametrisation']['spin-sph'])           ): continue
        if((key=='s1z')      and    (input_par['Waveform_and_parametrisation']['spin-sph'])           ): continue
        if((key=='s2x')      and    (input_par['Waveform_and_parametrisation']['spin-sph'])           ): continue
        if((key=='s2y')      and    (input_par['Waveform_and_parametrisation']['spin-sph'])           ): continue
        if((key=='s2z')      and    (input_par['Waveform_and_parametrisation']['spin-sph'])           ): continue
        if((key=='ecc')      and not(input_par['Waveform_and_parametrisation']['eccentricity'])       ): continue
        if(('lambda' in key) and not(input_par['Waveform_and_parametrisation']['tides'])              ): continue
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
    for key in default_test_values:
        
        if((key=='m1')       and    (input_par['Waveform_and_parametrisation']['mc-q-par'])           ): continue
        if((key=='m2')       and    (input_par['Waveform_and_parametrisation']['mc-q-par'])           ): continue
        if((key=='mc')       and not(input_par['Waveform_and_parametrisation']['mc-q-par'])           ): continue
        if((key=='q')        and not(input_par['Waveform_and_parametrisation']['mc-q-par'])           ): continue
        if((key=='s1s1')     and not(input_par['Waveform_and_parametrisation']['spin-sph'])           ): continue
        if((key=='s1s2')     and not(input_par['Waveform_and_parametrisation']['spin-sph'])           ): continue
        if((key=='s1s3')     and not(input_par['Waveform_and_parametrisation']['spin-sph'])           ): continue
        if((key=='s2s1')     and not(input_par['Waveform_and_parametrisation']['spin-sph'])           ): continue
        if((key=='s2s2')     and not(input_par['Waveform_and_parametrisation']['spin-sph'])           ): continue
        if((key=='s2s3')     and not(input_par['Waveform_and_parametrisation']['spin-sph'])           ): continue
        if((key=='s1x')      and    (input_par['Waveform_and_parametrisation']['spin-sph'])           ): continue
        if((key=='s1y')      and    (input_par['Waveform_and_parametrisation']['spin-sph'])           ): continue
        if((key=='s1z')      and    (input_par['Waveform_and_parametrisation']['spin-sph'])           ): continue
        if((key=='s2x')      and    (input_par['Waveform_and_parametrisation']['spin-sph'])           ): continue
        if((key=='s2y')      and    (input_par['Waveform_and_parametrisation']['spin-sph'])           ): continue
        if((key=='s2z')      and    (input_par['Waveform_and_parametrisation']['spin-sph'])           ): continue
    
        keytype = type(default_test_values[key])
        try:
            test_values[key]=keytype(Config.get('Test_values',key))
        except (KeyError, configparser.NoOptionError, TypeError):
            
            # Putting this block here allows to test the accuracy of the ROQ against regimes outside the training range (e.g. trained with tides=0 and checking the error with non-zero tides)
            if((key=='ecc')      and not(input_par['Waveform_and_parametrisation']['eccentricity'])       ): continue
            if(('lambda' in key) and not(input_par['Waveform_and_parametrisation']['tides'])              ): continue
            if((key=='s1x')      and not(input_par['Waveform_and_parametrisation']['spins']=='precessing')): continue
            if((key=='s2x')      and not(input_par['Waveform_and_parametrisation']['spins']=='precessing')): continue
            if((key=='s1y')      and not(input_par['Waveform_and_parametrisation']['spins']=='precessing')): continue
            if((key=='s2y')      and not(input_par['Waveform_and_parametrisation']['spins']=='precessing')): continue
            if((key=='s1z')      and    (input_par['Waveform_and_parametrisation']['spins']=='no-spins'  )): continue
            if((key=='s2z')      and    (input_par['Waveform_and_parametrisation']['spins']=='no-spins'  )): continue
            
            test_values[key]=default_test_values[key]
        if key in params_ranges.keys():
            if not(params_ranges[key][0] <= test_values[key] <= params_ranges[key][1]):
                warnings.warn("Chosen test value for {} outside training range.".format(key))

    return input_par, params_ranges, test_values
