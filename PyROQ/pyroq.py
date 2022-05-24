## -*- coding: utf8 -*-
#!/usr/bin/env python

# General python imports
import numpy as np, os, sys, random, time, warnings
from itertools import repeat

# Package internal imports
from .waveform_wrappers import *
from .parallel import eval_func_tuple
from . import initialise, linear_algebra, post_processing

# Initialize logger
import logging
logger = logging.getLogger(__name__)

# Inizialize error handlers
TermError    = ValueError('Unknown basis term requested.')
VersionError = ValueError('Unknown version requested.')
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

# ROQ main
class PyROQ:
    """
        PyROQ Class
    
        * Works with a list of waveform wrappers provided in `waveform_wrappers.py`.
    """

    def __init__(self,
                 config_pars                      ,
                 params_ranges                    ,
                 start_values               = None,
                 distance                   = 10  , # [Mpc]. Dummy value, distance does not enter the interpolants construction
                 additional_waveform_params = {}  , # Dictionary with any parameter needed for the waveform approximant
                 pool                       = None, # Parallel processing pool
                 ):

        self.distance                   = distance
        self.additional_waveform_params = additional_waveform_params
        self.params_ranges              = params_ranges
        self.start_values               = start_values
        
        # Read input params
        self.approximant                = config_pars['Waveform_and_parametrisation']['approximant']

        self.mc_q_par                   = config_pars['Waveform_and_parametrisation']['mc-q-par']
        self.spin_sph                   = config_pars['Waveform_and_parametrisation']['spin-sph']

        self.f_min                      = config_pars['Waveform_and_parametrisation']['f-min']
        self.f_max                      = config_pars['Waveform_and_parametrisation']['f-max']
        self.deltaF                     = 1./config_pars['Waveform_and_parametrisation']['seglen']
        
        self.tolerance_pre_basis_lin    = config_pars['ROQ']['tolerance-pre-basis-lin']
        self.tolerance_pre_basis_qua    = config_pars['ROQ']['tolerance-pre-basis-qua']
        self.n_pre_basis_search_iter    = config_pars['ROQ']['n-pre-basis-search-iter']
        self.n_pre_basis                = config_pars['ROQ']['n-pre-basis']

        self.n_training_set_cycles      = config_pars['ROQ']['n-training-set-cycles']
        self.training_set_sizes         = config_pars['ROQ']['training-set-sizes']
        self.training_set_n_outliers    = config_pars['ROQ']['training-set-n-outliers']
        self.training_set_rel_tol       = config_pars['ROQ']['training-set-rel-tol']
        self.tolerance_lin              = config_pars['ROQ']['tolerance-lin']
        self.tolerance_qua              = config_pars['ROQ']['tolerance-qua']

        self.n_tests_post               = config_pars['ROQ']['n-tests-post']
        self.error_version              = config_pars['ROQ']['error-version']
        self.minumum_speedup            = config_pars['ROQ']['minimum-speedup']

        self.parallel                   = config_pars['Parallel']['parallel']
        self.n_processes                = config_pars['Parallel']['n-processes']
        
        self.outputdir                  = config_pars['I/O']['output']
        self.timing                     = config_pars['I/O']['timing']
        
        # Set global pool object
        global Pool
        Pool = pool

        # Convert to LAL identification number, if passing a LAL approximant, and choose waveform
        from .waveform_wrappers import __non_lal_approx_names__
        if(not(config_pars['Waveform_and_parametrisation']['approximant'] in __non_lal_approx_names__)):
            self.approximant = lalsimulation.SimInspiralGetApproximantFromString(self.approximant)
        
        if self.approximant in WfWrapper.keys():
            self.wvf = WfWrapper[self.approximant](self.approximant, self.additional_waveform_params)
        else:
            raise ValueError('Unknown approximant requested.')

        # Build the map between params names and indexes
        self.map_params_indexs()  # Declares: self.i2n, self.n2i, self.nparams
        
        # Initial basis
        self.freq = np.arange(self.f_min, self.f_max, self.deltaF)
        self.set_training_range() # Declares: self.params_low, self.params_hig

    ## Parameters transformations utils

    def spherical_to_cartesian(self, sph):
        
        x = sph[0]*np.sin(sph[1])*np.cos(sph[2])
        y = sph[0]*np.sin(sph[1])*np.sin(sph[2])
        z = sph[0]*np.cos(sph[1])
        
        return [x,y,z]

    def get_m1m2_from_mcq(self, mc,q):
        
        m2 = mc * q ** (-0.6) * (1+q)**0.2
        m1 = m2 * q
        
        return np.array([m1,m2])

    def mass_range(self, mc_low, mc_high, q_low, q_high):
        
        mmin = self.get_m1m2_from_mcq(mc_low,q_high)[1]
        mmax = self.get_m1m2_from_mcq(mc_high,q_high)[0]
        
        return [mmin, mmax]

    ## Parameters handling functions

    def map_params_indexs(self):
        
        """
            Build a map between the parameters names and the indexes of the parameter arrays, and its inverse.
        """
        names = self.params_ranges.keys()
        self.nparams = len(names)
        self.n2i = dict(zip(names,range(self.nparams)))
        self.i2n = {i: n for n, i in self.n2i.items()}
        
        return

    def update_waveform_params(self, paramspoint):
        
        """
            Update the waveform parameters (dictionary) with those in paramspoint (np.array).
        """
        p = {}
        for i,k in self.i2n.items():
            p[k] = paramspoint[i]

        return p
         
    def generate_params_points(self, npts, round_to_digits=6):
        
        """
            Uniformly sample the parameter arrays
        """
        paramspoints = np.random.uniform(self.params_low,
                                         self.params_hig,
                                         size=(npts, self.nparams))
                                         
        return paramspoints.round(decimals=round_to_digits)
    
    def paramspoint_to_wave(self, paramspoint, term):
        
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
    
        # We build a linear basis only for hp, since empirically the same basis accurately works to represent hc too (see [arXiv:1604.08253]).
        hp, hc = self.wvf.generate_waveform(p, self.deltaF, self.f_min, self.f_max, self.distance)
        
        if   term == 'lin': pass
        elif term == 'qua': hp, hc = (np.absolute(hp))**2, (np.absolute(hc))**2
        else              : raise TermError
        
        return hp, hc

    ## Basis construction functions

    def add_new_element_to_basis(self, new_basis_param_point, known_basis, known_params, term):
        
        # Create new basis element.
        hp_new, _ = self.paramspoint_to_wave(new_basis_param_point, term)

        # Orthogonalise, i.e. extract the linearly independent part of the waveform, and normalise the new element, which constitutes a new basis element. Note: the new basis element is not a 'waveform', since subtraction of two waveforms does not generate a waveform.
        basis_new = linear_algebra.gram_schmidt(known_basis, hp_new, self.deltaF)

        # Append to basis.
        known_basis  = np.append(known_basis,  np.array([basis_new]),             axis=0)
        known_params = np.append(known_params, np.array([new_basis_param_point]), axis=0)

        return known_basis, known_params

    def compute_new_element_residual_modulus_from_basis(self, paramspoint, known_basis, term):

        # Create and normalise element to be projected and initialise residual
        h_to_proj, _ = self.paramspoint_to_wave(paramspoint, term)
        h_to_proj    = linear_algebra.normalise_vector(h_to_proj, self.deltaF)
        residual     = h_to_proj

        #FIXME: this block has a large repetition with `gram_schmidt`, except for norm.
        # Subtract the projection on the basis from the residual.
        for k in np.arange(0,len(known_basis)):
            residual -= linear_algebra.projection(known_basis[k],h_to_proj)
        
        return linear_algebra.scalar_product(residual, residual, self.deltaF)
        
    def search_new_basis_element(self, paramspoints, known_basis, term):

        """
           Given an array of new random points in the parameter space (paramspoints) and the known basis elements, this function searches and constructs a new basis element. The new element is constructed by:
           
           1) Projecting the waveforms corresponding to parampoints on the known basis;
           2) Selecting the waveform with the largest residual (modulus) after projection;
           3) Computing the normalised residual projection of the selected waveform on the known basis.
        """

        # Generate len(paramspoints) random waveforms corresponding to parampoints.
        modula           = list(Pool.map(eval_func_tuple, zip(repeat(self.compute_new_element_residual_modulus_from_basis),
                                                              paramspoints,
                                                              repeat(known_basis),
                                                              repeat(term))))

        # Select the worst represented waveform (in terms of the previous known basis).
        arg_newbasis = np.argmax(modula) 
       
        return paramspoints[arg_newbasis], modula[arg_newbasis]
            
    def construct_preselection_basis(self, known_basis, params, residual_modula, term):
        
        if term == 'lin':
            file_basis    = self.outputdir+'/ROQ_data/Linear/preselection_linear_basis.npy'
            file_params   = self.outputdir+'/ROQ_data/Linear/preselection_linear_basis_waveform_params.npy'
            tolerance_pre = self.tolerance_pre_basis_lin
        elif term=='qua':
            file_basis    = self.outputdir+'/ROQ_data/Quadratic/preselection_quadratic_basis.npy'
            file_params   = self.outputdir+'/ROQ_data/Quadratic/preselection_quadratic_basis_waveform_params.npy'
            tolerance_pre = self.tolerance_pre_basis_qua
        else:
            raise TermError
    
        # This block generates a basis of dimension n_pre_basis.
        logger.info('')
        logger.info('#############################')
        logger.info('# \u001b[38;5;\u001b[38;5;39mStarting {} preselection\u001b[0m #'.format(term))
        logger.info('#############################')
        logger.info('')
        logger.info('N points per iter  : {}'.format(self.n_pre_basis_search_iter))
        logger.info('Tolerance          : {}'.format(tolerance_pre))
        logger.info('Maximum iterations : {}'.format(self.n_pre_basis-2)) # The -2 comes from the fact that the corner basis is composed by two elements.
        logger.info('')

        k = 0
        while(residual_modula[-1] > tolerance_pre):
            
            # Generate n_pre_basis_search_iter random points.
            paramspoints = self.generate_params_points(self.n_pre_basis_search_iter)
            
            if(self.timing): execution_time_new_pre_basis_element = time.time()
            # From the n_pre_basis_search_iter randomly generated points, select the worst represented waveform corresponding to that point (i.e. with the largest residuals after basis projection).
            params_new, rm_new = self.search_new_basis_element(paramspoints, known_basis, term)
            if(self.timing): logger.info('Timing: pre-selection basis {} iteration, generating {} waveforms with parallel={} [minutes]: {}'.format(k+1, self.n_pre_basis_search_iter, self.parallel, (time.time() - execution_time_new_pre_basis_element)/60.0))
            logger.info('Pre-selection iteration: {}'.format(k+1) + ' -- Largest projection error: {}'.format(rm_new))

            # The worst represented waveform becomes the new basis element.
            known_basis, params = self.add_new_element_to_basis(params_new, known_basis, params, term)
            residual_modula     = np.append(residual_modula, rm_new)

            # If a maximum number of iterations was given, stop at that number, otherwise continue until tolerance is reached.
            if((self.n_pre_basis > 2) and (len(known_basis[:,0]) >= self.n_pre_basis)): break
            else                                                                      : k = k+1
                
        # Store the pre-selected basis.
        np.save(file_basis,  known_basis)
        np.save(file_params, params     )

        return known_basis, params, residual_modula
    
    ## Initial basis functions
    
    def set_training_range(self):
        """
            Initialize parameter ranges and basis.
        """
        
        logger.info('######################')
        logger.info('# \u001b[\u001b[38;5;39mInitialising basis\u001b[0m #')
        logger.info('######################')
        logger.info('')
        logger.info('nparams = {}'.format(self.nparams))
        logger.info('')
        logger.info('index | name    | ( min - max )           ')

        self.params_low, self.params_hig = [], []
        # Set bounds
        for i,n in self.i2n.items():
            self.params_low.append(self.params_ranges[n][0])
            self.params_hig.append(self.params_ranges[n][1])
            
            logger.info('{}    | {} | ( {:.6f} - {:.6f} ) '.format(str(i).ljust(2), n.ljust(len('lambda1')), self.params_low[i], self.params_hig[i]))
        logger.info('')
        
        return 

    def construct_corner_basis(self, term):

        hp_low, _ = self.paramspoint_to_wave(self.params_low, term)
        
        # Initialise the base with the lowest corner.
        known_basis_start     = np.array([linear_algebra.normalise_vector(hp_low, self.deltaF)])
        params_ini            = np.array([self.params_low])
        residual_modula_start = np.array([1.0])

        # Add the highest corner.
        known_basis_start, params_ini = self.add_new_element_to_basis(self.params_hig, known_basis_start, params_ini, term)
        residual_modula_start         = np.append(residual_modula_start, np.array([1.0]))

        return known_basis_start, params_ini, residual_modula_start

    ## Interpolant building functions

    def compute_empirical_interpolation_error(self, training_point, basis_interpolant, emp_nodes, term):

        # Create and normalise benchmark waveform.
        hp, _ = self.paramspoint_to_wave(training_point, term)
        hp    = linear_algebra.normalise_vector(hp, self.deltaF)

        # Compute the empirical interpolation error.
        hp_interp = np.dot(basis_interpolant,hp[emp_nodes])
        dh        = hp - hp_interp

        return linear_algebra.scalar_product(dh, dh, self.deltaF)

    def search_worst_represented_point(self, outliers, basis_interpolant, emp_nodes, training_set_tol, term):
        
        execution_time_search_worst_point = time.time()
        # Loop over test points.
        eies = list(Pool.map(eval_func_tuple, zip(repeat(self.compute_empirical_interpolation_error),
                                                  outliers,
                                                  repeat(basis_interpolant),
                                                  repeat(emp_nodes),
                                                  repeat(term))))
                                                  
        if(self.timing):
            execution_time_search_worst_point = (time.time() - execution_time_search_worst_point)/60.0
            logger.info('Timing: worst point search, computing {} interpolation errors with parallel={} [minutes]: {}'.format(len(outliers), self.parallel, execution_time_search_worst_point))

        # Select the worst represented point.
        arg_worst                     = np.argmax(eies)
        worst_represented_param_point = outliers[arg_worst]
        
        # Update outliers.
        outliers = outliers[np.array(eies) > training_set_tol]

        return worst_represented_param_point, eies[arg_worst], outliers

    def empirical_nodes(self, known_basis):
        
        """
            Generate the empirical interpolation nodes from a given basis.
            Follows the algorithm detailed in Ref. Phys. Rev. X 4, 031006, according to PRD 104, 063031 (2021).
            See also arXiv:1712.08772v2 for a description.
        """
        
        # Initialise. The first point is chosen to maximise the first basis vector.
        basis_len = len(known_basis)
        emp_nodes = np.array([np.argmax(np.absolute(known_basis[0]))])
        
        # The second point is chosen to maximise the difference between the interpolant (constructed from the first basis vector) and the second basis vector.
        c1        = known_basis[1,emp_nodes[0]]/known_basis[0,1]
        interp1   = np.multiply(c1,known_basis[0])
        diff1     = interp1 - known_basis[1]
        r1        = np.absolute(diff1)
        emp_nodes = np.append(emp_nodes, np.argmax(r1))
        # Make sure frequencies are ordered.
        emp_nodes = sorted(emp_nodes)

        # Then iterate for all the other nodes.
        for k in np.arange(2,basis_len):
            
            Vtmp         = np.transpose(known_basis[0:k,emp_nodes])
            inverse_Vtmp = np.linalg.pinv(Vtmp)
            e_to_interp  = known_basis[k]
            Ci           = np.dot(inverse_Vtmp, e_to_interp[emp_nodes])
            interpolantA = np.zeros(len(known_basis[k]))+np.zeros(len(known_basis[k]))*1j
            
            for j in np.arange(0, k):
                tmp           = np.multiply(Ci[j], known_basis[j])
                interpolantA += tmp
            
            diff         = interpolantA - known_basis[k]
            r            = np.absolute(diff)
            emp_nodes    = np.append(emp_nodes, np.argmax(r))
            emp_nodes    = sorted(emp_nodes)

        # Remove repetitions, otherwise duplicates on the frequency axis will bias likelihood computation during parameter estimation.
        emp_nodes = np.unique(emp_nodes)
        ndim      = len(emp_nodes)
        V         = np.transpose(known_basis[0:ndim, emp_nodes])
        inverse_V = np.linalg.pinv(V)
        
        if not(ndim==basis_len): logger.info('Removed {} duplicate points during empirical interpolation nodes construction.'.format(basis_len-ndim))
        
        return np.array([ndim, inverse_V, emp_nodes])
    
    def roqs(self, known_basis, known_params, term):

        # Initialise iteration and create paths in which to store the output.
        if term == 'lin':
            tol                  = self.tolerance_lin
            file_interpolant     = self.outputdir+'/ROQ_data/Linear/basis_interpolant_linear.npy'
            file_empirical_freqs = self.outputdir+'/ROQ_data/Linear/empirical_frequencies_linear.npy'
            file_basis           = self.outputdir+'/ROQ_data/Linear/basis_linear.npy'
            file_params          = self.outputdir+'/ROQ_data/Linear/basis_waveform_params_linear.npy'
        elif term == 'qua':
            tol                  = self.tolerance_qua
            file_interpolant     = self.outputdir+'/ROQ_data/Quadratic/basis_interpolant_quadratic.npy'
            file_empirical_freqs = self.outputdir+'/ROQ_data/Quadratic/empirical_frequencies_quadratic.npy'
            file_basis           = self.outputdir+'/ROQ_data/Quadratic/basis_quadratic.npy'
            file_params          = self.outputdir+'/ROQ_data/Quadratic/basis_waveform_params_quadratic.npy'
        else:
            raise TermError

        maximum_eies, n_outliers = np.array([]), np.array([])
        # Start a loop over training cycles with varying training size, tolerance and number of allowed outliers.
        for n_cycle in range(self.n_training_set_cycles):
            
            training_set_size      = self.training_set_sizes[n_cycle]
            training_set_n_outlier = self.training_set_n_outliers[n_cycle]
            training_set_tol       = self.training_set_rel_tol[n_cycle] * tol
        
            logger.info('')
            logger.info('')
            logger.info('################################')
            logger.info('# \u001b[\u001b[38;5;39mStarting {}/{} enrichment loop\u001b[0m #'.format(n_cycle+1, self.n_training_set_cycles))
            logger.info('################################')
            logger.info('')
            logger.info('Training set size  : {}'.format(training_set_size))
            logger.info('Tolerance          : {}'.format(training_set_tol))
            logger.info('Tolerated outliers : {}'.format(training_set_n_outlier))
            logger.info('')

            # Generate the parameters of this training cycle.
            paramspoints = self.generate_params_points(npts=training_set_size)
            outliers     = paramspoints[:training_set_size]

            while(len(outliers) > training_set_n_outlier):

                # Store the current basis and parameters at each step, as a backup.
                np.save(file_basis,  known_basis)
                np.save(file_params, known_params)

                # From the basis constructed above, extract: the empirical interpolation nodes (i.e. the subset of frequencies on which the ROQ rule is evaluated); the basis interpolant, which allows to construct an arbitrary waveform at an arbitrary frequency point from the constructed basis.
                ndim, inverse_V, emp_nodes = self.empirical_nodes(known_basis)
                basis_interpolant          = np.dot(np.transpose(known_basis[0:ndim]),inverse_V)
                
                # Out of the remaining outliers, select the worst represented point.
                worst_represented_param_point, maximum_eie, outliers = self.search_worst_represented_point(outliers, basis_interpolant, emp_nodes, training_set_tol, term)

                # Update the user on how many outliers remain.
                logger.info('{}'.format(ndim)+' basis elements gave {} outliers with interpolation error > {} out of {} training points.'.format(len(outliers), training_set_tol, training_set_size))
                logger.info('Largest interpolation error: {}'.format(maximum_eie))

                # Enrich the basis with the worst outlier. Also store the maximum empirical interpolation error, to monitor the improvement in the interpolation.
                if(len(outliers) > 0):
                    known_basis, known_params = self.add_new_element_to_basis(worst_represented_param_point, known_basis, known_params, term)
                    maximum_eies              = np.append(maximum_eies, maximum_eie)
                    n_outliers                = np.append(n_outliers  , len(outliers))

                # Check if basis construction became pointless.
                if((len(self.freq)/len(known_basis[:,0])) < self.minumum_speedup): raise Exception('Basis dimension is larger than the minimum speedup requested. Aborting the interpolants construction.')

        # Finalise and store the output.
        frequencies = self.freq[emp_nodes]
        np.save(file_interpolant,     basis_interpolant)
        np.save(file_empirical_freqs, frequencies)
        
        return frequencies, basis_interpolant, known_params, maximum_eies, n_outliers

    ## Main function handling the ROQ construction.

    def run(self, term):
        
        # Initialise data.
        d = {}

        # Initialise basis, either using a previously constructed one or pre-selecting one from corners of the parameter space plus a user-determined number of iterations..
        execution_time_presel_basis = time.time()
        if(self.start_values==None):
            # We choose the first elements of the basis to correspond to the lower and upper values of the parameters range. Note that corner does not mean the N-D corners of the parameter space N-cube, but simply upper-lower bounds.
            initial_basis, initial_params, initial_residual_modula = self.construct_corner_basis(term)
            # Run a first pre-selection loop, building a basis of dimension `n_pre_basis`.
            preselection_basis, preselection_params, preselection_residual_modula = self.construct_preselection_basis(initial_basis, initial_params, initial_residual_modula, term)
        else:
            # FIXME: load a previously constructed basis.
            preselection_basis, preselection_params, preselection_residual_modula = None, None, None
            raise Exception('User-input initial basis has not been implemented yet.')
        if(self.timing):
            execution_time_presel_basis = (time.time() - execution_time_presel_basis)/60.0
            logger.info('Timing: pre-selection basis with parallel={} [minutes]: {}'.format(self.parallel, execution_time_presel_basis))

        # Internally store the output data for later testing.
        d['{}_pre_basis'.format(term)]   = preselection_basis
        d['{}_pre_params'.format(term)]  = preselection_params
        d['{}_pre_res_mod'.format(term)] = preselection_residual_modula

        # Start the series of loops in which the pre-selected basis is enriched by the outliers found on ever increasing training sets.
        frequencies, basis_interpolant, basis_parameters, maximum_eies, n_outliers = self.roqs(preselection_basis, preselection_params, term)

        # Internally store the output data for later testing.
        d['{}_interpolant'.format(term)] = basis_interpolant
        d['{}_f'.format(term)]           = frequencies
        d['{}_emp_nodes'.format(term)]   = np.searchsorted(self.freq, d['{}_f'.format(term)])
        d['{}_params'.format(term)]      = basis_parameters
        d['{}_max_eies'.format(term)]    = maximum_eies
        d['{}_n_outliers'.format(term)]  = n_outliers

        return d
