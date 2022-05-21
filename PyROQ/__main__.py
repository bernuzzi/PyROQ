## -*- coding: utf8 -*-
#!/usr/bin/env python

# General python imports
import multiprocessing as mp, numpy as np, os, sys, random, time, warnings
from optparse import OptionParser
from itertools import repeat

# Package internal imports
from . import initialise, post_processing

# Initialize logger
import logging

# Inizialize error handlers
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

# Logger setter
def set_logger(label=None, outdir=None, level='INFO', verbose=True):
    
    # Set formatters
    datefmt = '%m-%d-%Y %H:%M'
    fmt     = '[{}] [%(asctime)s] %(message)s'.format(label)

    # Initialize logger
    logger = logging.getLogger(label)
    logger.propagate = False
    logger.setLevel(('{}'.format(level)).upper())

    # Set stream-handler (i.e. console)
    if verbose:
        if any([type(h) == logging.StreamHandler for h in logger.handlers]) is False:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
            stream_handler.setLevel(('{}'.format(level)).upper())
            logger.addHandler(stream_handler)
    
    # Set file-handler (i.e. file)
    if outdir != None:
        if any([type(h) == logging.FileHandler for h in logger.handlers]) is False:
            log_file = '{}/{}.log'.format(outdir, label)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
            file_handler.setLevel(('{}'.format(level)).upper())
            logger.addHandler(file_handler)

    return logger

if __name__ == '__main__':

    # Initialise and read config.
    parser      = OptionParser(initialise.usage)
    parser.add_option('--config-file',  type='string',          metavar = 'config_file',    default = None)
    parser.add_option('--debug',        action = 'store_true',  metavar = 'debug',          default = False)
    (opts,args) = parser.parse_args()
    config_file = opts.config_file
    
    config_pars, params_ranges, test_values = initialise.read_config(config_file)
    
    # set logger(s)
    if opts.debug:
        logger =    set_logger(label='PyROQ',
                               level='DEBUG',
                               outdir=config_pars['I/O']['output'],
                               verbose=bool(config_pars['I/O']['verbose']),)
    else:
        logger =    set_logger(label='PyROQ',
                               outdir=config_pars['I/O']['output'],
                               verbose=bool(config_pars['I/O']['verbose']),)
    
    # Get parallel processing pool
    if (int(config_pars['Parallel']['parallel'])==0):
        logger.info('Initialising serial pool')
        from .parallel import initialize_serial_pool
        Pool = initialize_serial_pool()
    elif (int(config_pars['Parallel']['parallel'])==1):
        logger.info('Initialising multiprocessing processsing pool')
        from .parallel import initialize_mp_pool, close_pool_mp
        Pool = initialize_mp_pool(int(config_pars['Parallel']['n-processes']))
        close_pool = close_pool_mp
    elif (int(config_pars['Parallel']['parallel'])==2):
        logger.info('Initialising MPI-based processing pool')
        from .parallel import initialize_mpi_pool, close_pool_mpi
        Pool = initialize_mpi_pool()
        close_pool = close_pool_mpi
    else:
        raise ValueError("Unable to initialise parallelisation method. Use parallel=0 for serial, parallel=1 for multiprocessing or parallel=2 for MPI.")

    # Set random seed
    if (int(config_pars['Parallel']['parallel'])<2):
        logger.info('Setting random seed to {}'.format(config_pars['I/O']['random-seed']))
        np.random.seed(int(config_pars['I/O']['random-seed']))
    else:
        # Avoid generation of identical random numbers in different processes
        if Pool.is_master():
            logger.info('Setting random seed to {}'.format(config_pars['I/O']['random-seed']))
            np.random.seed(int(config_pars['I/O']['random-seed']))
        else:
            np.random.seed(int(config_pars['I/O']['random-seed'])+Pool.rank)

    # Open pool
    with Pool as pool:
    
        # If MPI, set workers in wait for commands from master
        if (int(config_pars['Parallel']['parallel'])==2):
            if not pool.is_master():
                pool.wait()
                sys.exit(0)

        # Point(s) of the parameter space on which to initialise the basis.
        # If not passed by the user, defaults to upper/lower corner of the parameter space.
        start_values = None
        
#        # Set random seed
#        if config_pars['I/O']['random-seed']:
#            logger.info('Setting random seed to {}'.format(config_pars['I/O']['random-seed']))
#            np.random.seed(int(config_pars['I/O']['random-seed']))
#        else:
#            np.random.seed(0)

        # Initialise ROQ parameters and structures.
        from . import pyroq as roq
        pyroq = roq.PyROQ(config_pars, params_ranges, start_values=start_values, pool=pool)
        freq  = pyroq.freq

        data = {}

        for run_type in config_pars['I/O']['run-types']:

            term = run_type[0:3]
            if not(config_pars['I/O']['post-processing-only']):
                # Create the basis and save ROQ.
                data[run_type] = pyroq.run(term)
            
                # These data are not saved in output, so plot them now.
                post_processing.plot_preselection_residual_modula(data[run_type]['{}_pre_res_mod'.format(term)], term, pyroq.outputdir)
                post_processing.plot_maximum_empirical_interpolation_error(data[run_type]['{}_max_eies'.format(term)], term, pyroq.outputdir)
                post_processing.plot_number_of_outliers(data[run_type]['{}_n_outliers'.format(term)], term, pyroq.outputdir)

            else:
                # Read ROQ from previous run.
                data[run_type]                                = {}
                data[run_type]['{}_f'.format(term)]           = np.load(os.path.join(config_pars['I/O']['output'],'ROQ_data/{type}/empirical_frequencies_{type}.npy'.format(type=run_type)))
                data[run_type]['{}_interpolant'.format(term)] = np.load(os.path.join(config_pars['I/O']['output'],'ROQ_data/{type}/basis_interpolant_{type}.npy'.format(type=run_type)))
                data[run_type]['{}_emp_nodes'.format(term)]   = np.searchsorted(freq, data[run_type]['{}_f'.format(term)])
                data[run_type]['{}_params'.format(term)]      = np.load(os.path.join(config_pars['I/O']['output'],'ROQ_data/{type}/basis_waveform_params_{type}.npy'.format(type=run_type)))

            # Output the basis reduction factor.
            logger.info('Results')
            logger.info('{} basis reduction factor: (Original freqs [{}]) / (New freqs [{}]) = {}'.format(run_type, len(freq), len(data[run_type]['{}_f'.format(term)]), len(freq)/len(data[run_type]['{}_f'.format(term)])))

            # Plot the basis parameters corresponding to the selected basis (only the first N elements determined during the interpolant construction procedure).
            post_processing.histogram_basis_params(data[run_type]['{}_params'.format(term)][:len(data[run_type]['{}_f'.format(term)])], pyroq.outputdir, pyroq.i2n)

            # Surrogate tests.
            post_processing.test_roq_error(data[run_type]['{}_interpolant'.format(term)], data[run_type]['{}_emp_nodes'.format(term)], term, pyroq)

            # Plot the representation error for a random waveform, using the interpolant built from the constructed basis. Useful for visual diagnostics.
            logger.info('Testing the waveform using the parameters:')
            parampoint_test = []
            logger.info('name    | value | index')
            for name, val in test_values.items():
                logger.info('{} | {}   | {} '.format(name.ljust(len('lambda1')), val, pyroq.n2i[name]))
                parampoint_test.append(val)
            parampoint_test = np.array(parampoint_test)

            post_processing.plot_representation_error(data[run_type]['{}_interpolant'.format(term)], data[run_type]['{}_emp_nodes'.format(term)], parampoint_test, term, pyroq.outputdir, freq, pyroq.paramspoint_to_wave)

    # Show plots, if requested.
    if(config_pars['I/O']['show-plots']): plt.show()