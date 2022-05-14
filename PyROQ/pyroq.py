# General python imports
import multiprocessing as mp, numpy as np, os, random, time, warnings
from optparse import OptionParser

# Package internal import
from wvfwrappers    import *
from linear_algebra import *
import initialise, post_processing

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
np.set_printoptions(linewidth=np.inf)
TermError    = ValueError("Unknown basis term requested.")
VersionError = ValueError("Unknown version requested.")
np.random.seed(150914)

class PyROQ:
    """
    PyROQ Class
    
    * Works with a list of very basic waveform wrappers provided in wvfwrappers.py
    
    * The parameter space is *defined* by the keywords of 'params_ranges' 

    """

    def __init__(self,
                 config_pars                      ,
                 params_ranges                    ,
                 start_values               = None,
                 distance                   = 10  , # [Mpc]. Dummy value, distance does not enter the interpolants construction
                 additional_waveform_params = {}  , # Dictionary with any parameter needed for the waveform approximant
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

        self.parallel                   = config_pars['Parallel']['parallel']
        self.n_processes                = config_pars['Parallel']['n-processes']
        
        self.outputdir                  = config_pars['I/O']['output']
        self.verbose                    = config_pars['I/O']['verbose']
        self.timing                     = config_pars['I/O']['timing']

        # Convert to LAL identification number, if passing a LAL approximant, and choose waveform
        if(not(config_pars['Waveform_and_parametrisation']['approximant']=='teobresums-giotto') and not(config_pars['Waveform_and_parametrisation']['approximant']=='mlgw-bns')):
            self.approximant = lalsimulation.SimInspiralGetApproximantFromString(self.approximant)
        if self.approximant in WfWrapper.keys():
            self.wvf = WfWrapper[self.approximant](self.approximant, self.additional_waveform_params)
        else:
            raise ValueError('Unknown approximant requested.')

        # Build the map between params names and indexes
        self.map_params_indexs()  # self.i2n, self.n2i, self.nparams
        
        # Initial basis
        self.freq = np.arange(self.f_min, self.f_max, self.deltaF)
        self.set_training_range() # self.params_low, self.params_hig

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

    def mass_range(self,mc_low, mc_high, q_low, q_high):
        mmin = self.get_m1m2_from_mcq(mc_low,q_high)[1]
        mmax = self.get_m1m2_from_mcq(mc_high,q_high)[0]
        return [mmin, mmax]

    ## Parameters handling functions

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
         
    def generate_params_points(self,npts,round_to_digits=6):
        """
        Uniformly sample the parameter arrays
        """
        paramspoints = np.random.uniform(self.params_low,
                                         self.params_hig,
                                         size=(npts,self.nparams))
        return paramspoints.round(decimals=round_to_digits)
    
    def paramspoint_to_wave(self, paramspoint):
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

    ## Basis construction functions

    def _compute_new_element_residual_from_basis(self, paramspoint, known_bases, term):

        #FIXME: this function has a large repetition with gram_schmidt
        hp, _ = self.paramspoint_to_wave(paramspoint)
        if   term == 'lin': residual = hp
        elif term == 'qua': residual = (np.absolute(hp))**2
        else              : raise TermError
        h_to_proj = residual

        for k in np.arange(0,len(known_bases)):
            residual -= proj(known_bases[k],h_to_proj)
        
        return np.sqrt(np.vdot(residual, residual))
        
    def _search_new_basis_element(self, paramspoints, known_bases, term):

        """

           Given an array of new random points in the parameter space (paramspoints) and the known basis elements, this function searches and constructs a new basis element. The new element is constructed by:
           
           1) Projecting the waveforms corresponding to parampoints on the known basis;
           2) Selecting the waveform with the largest residual (modulus) after projection;
           3) Computing the normalised residual projection of the selected waveform on the known basis.
           
        """

        # Generate len(paramspoints) random waveforms corresponding to parampoints.
        if self.parallel:
            paramspointslist = paramspoints.tolist()
            pool = mp.Pool(processes=self.n_processes)
            modula = [pool.apply(self._compute_new_element_residual_from_basis, args=(paramspoint, known_bases, term)) for paramspoint in paramspointslist]
            pool.close()
        else:
            npts   = len(paramspoints)
            modula = np.zeros(npts)
            for i,paramspoint in enumerate(paramspoints):
                modula[i] = np.real(self._compute_new_element_residual_from_basis(paramspoint, known_bases, term))

        # Select the worst represented waveform (in terms of the previous known basis).
        arg_newbasis = np.argmax(modula) 
        hp, _        = self.paramspoint_to_wave(paramspoints[arg_newbasis])
        if   term == 'lin': pass
        elif term == 'qua': hp = (np.absolute(hp))**2
        else              : raise TermError
        
        # Extract the linearly independent part of the worst represented waveform, which constitutes a new basis element.
        # Note: the new basis element is not a 'waveform', since subtraction of two waveforms does not generate a waveform.
        basis_new = gram_schmidt(known_bases, hp)
       
        return np.array([basis_new, paramspoints[arg_newbasis], modula[arg_newbasis]])
            
    def _construct_preselection_basis(self, known_bases, params, residual_modula, term):
        
        if term == 'lin':
            file_bases  = self.outputdir+'/ROQ_data/Linear/preselection_linear_bases.npy'
            file_params = self.outputdir+'/ROQ_data/Linear/preselection_linear_bases_waveform_params.npy'
        elif term=='qua':
            file_bases  = self.outputdir+'/ROQ_data/Quadratic/preselection_quadratic_bases.npy'
            file_params = self.outputdir+'/ROQ_data/Quadratic/preselection_quadratic_bases_waveform_params.npy'
        else:
            raise TermError
    
        # This block generates a basis of dimension n_pre_basis.
        print('\n\n###################################################################################\n# Starting preselection {} iteration (with {} random points at each iteration) #\n###################################################################################\n'.format(term, self.n_pre_basis_search_iter))
        # The -2 comes from the fact that the corner basis is composed by two elements.
        total_iters = self.n_pre_basis-2
        for k in np.arange(0, total_iters):
            
            # Generate n_pre_basis_search_iter random points.
            paramspoints = self.generate_params_points(self.n_pre_basis_search_iter)
            
            # From the n_pre_basis_search_iter randomly generated points, select the worst represented waveform corresponding to that point (i.e. with the largest residuals after basis projection).
            execution_time_new_pre_basis_element = time.time()
            basis_new, params_new, rm_new = self._search_new_basis_element(paramspoints, known_bases, term)
            if(self.timing):
                execution_time_new_pre_basis_element = (time.time() - execution_time_new_pre_basis_element)/60.0
                print('Timing: pre-selection basis {} iteration, generating {} waveforms with parallel={} [minutes]: {}'.format(k+1, self.n_pre_basis_search_iter, self.parallel, execution_time_new_pre_basis_element))
            if self.verbose:
                np.set_printoptions(suppress=True)
                print("Preselection iteration: {}/{}".format(k+1, total_iters), " -- New basis waveform with parameters:", params_new)
                np.set_printoptions(suppress=False)

            # The worst represented waveform becomes the new basis element.
            known_bases     = np.append(known_bases,     np.array([basis_new]),  axis=0)
            params          = np.append(params,          np.array([params_new]), axis=0)
            residual_modula = np.append(residual_modula, rm_new)

        # Store the pre-selected basis.
        np.save(file_bases,  known_bases)
        np.save(file_params, params     )

        return known_bases, params, residual_modula
    
    ## Initial basis functions
    
    def set_training_range(self):
        """
        Initialize parameter ranges and basis.
        """
        if self.verbose:
            print('\n\n######################\n# Initialising basis #\n######################\n')
            print('nparams = {}\n'.format(self.nparams))
            print('index | name    | ( min - max )           ')

        self.params_low, self.params_hig = [], []
        # Set bounds
        for i,n in self.i2n.items():
            self.params_low.append(self.params_ranges[n][0])
            self.params_hig.append(self.params_ranges[n][1])
            
            if self.verbose: print('{}    | {} | ( {:.6f} - {:.6f} ) '.format(str(i).ljust(2), n.ljust(len('lambda1')), self.params_low[i], self.params_hig[i]))

        return 

    def _construct_corner_basis(self, run_type):

        # Corner waveforms
        self.hp_low, _ = self.paramspoint_to_wave(self.params_low)
        self.hp_hig, _ = self.paramspoint_to_wave(self.params_hig)
        if  (run_type=='lin'): hp_low, hp_hig = self.hp_low, self.hp_hig
        elif(run_type=='qua'): hp_low, hp_hig = (np.absolute(self.hp_low))**2, (np.absolute(self.hp_hig))**2
        else                 : raise TermError
        # FIXME: should test if it's more efficient to gram_schmidt hp2 before adding it to the basis.
        known_bases_start = np.array([vector_normalised(hp_low)])
        known_bases_start = np.append(known_bases_start, np.array([vector_normalised(hp_hig)]), axis=0)
        
        # Corner params
        params_ini = np.array([self.params_low])
        params_ini = np.append(params_ini, np.array([self.params_hig]), axis=0)
        
        # Corner residuals
        residual_modula_start = np.array([0.0])
        residual_modula_start = np.append(residual_modula_start, np.array([0.0]))

        return known_bases_start, params_ini, residual_modula_start

    ## Interpolant building functions

    def empirical_nodes(self, ndim, known_bases, fact=100000000):
        
        """
        Generate the empirical interpolation nodes from a given basis.
        Follows the algorithm detailed in Ref. Phys. Rev. X 4, 031006, according to PRD 104, 063031 (2021).
        See also arXiv:1712.08772v2 for a description.
        """
        
        # Initialise.
        # FIXME: why do we need to multiply by `fact`?
        emp_nodes    = np.arange(0,ndim) * fact
        
        # The first point is chosen to maximise the first basis vector.
        emp_nodes[0] = np.argmax(np.absolute(known_bases[0]))
        
        # The second point is chosen to maximise the difference between the interpolant (constructed from the first basis vector) and the second basis vector.
        c1           = known_bases[1,emp_nodes[0]]/known_bases[0,1]
        interp1      = np.multiply(c1,known_bases[0])
        diff1        = interp1 - known_bases[1]
        r1           = np.absolute(diff1)
        emp_nodes[1] = np.argmax(r1)
        
        # Then iterate for all the other nodes.
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
    
    def _roqs(self, known_bases, known_params, term):

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

        # Start a loop over training cycles with varying training size, tolerance and allowed outliers
        for n_cycle in range(self.n_training_set_cycles):
            
            training_set_size      = self.training_set_sizes[n_cycle]
            training_set_n_outlier = self.training_set_n_outliers[n_cycle]
            training_set_tol       = self.training_set_rel_tol[n_cycle] * tol
        
            print('\n################################\n# Starting {}/{} enrichment loop #\n################################\n\nTraining set size  : {}\nTolerance          : {}\nTolerated outliers : {}\n\n'.format(n_cycle+1, self.n_training_set_cycles, training_set_size, training_set_tol, training_set_n_outlier))

            paramspoints = self.generate_params_points(npts=training_set_size)
            outliers     = paramspoints[ :training_set_size]
            
            while(len(outliers) > training_set_n_outlier):

                np.save(file_basis,  known_bases)
                np.save(file_params, known_params)

                # From the basis constructed above, extract:
                # 1) the empirical interpolation nodes (i.e. the subset of frequencies on which the ROQ rule is evaluated);
                # 2) the basis interpolant, which allows to construct an arbitrary waveform at an arbitrary frequency point from the constructed basis.
                ndim, inverse_V, emp_nodes = self.empirical_nodes(len(known_bases), known_bases)
                if(ndim>=len(self.freq)): raise Exception('Basis dimension is equal or larger than original frequency points, hence ROQ will not speedup likelihood evaluations. Try decreasing the tolerance or improving basis construction strategy.')
                basis_interpolant          = np.dot(np.transpose(known_bases[0:ndim]),inverse_V)
                print('FIXME: PARALLELISE ME!\n')
                
                # Initialise empirical interpolation errors and loop over test points.
                eies = []
                for training_point in outliers:

                    # Create benchmark waveform.
                    hp, _ = self.paramspoint_to_wave(training_point)
                    if   term == 'lin': pass
                    elif term == 'qua': hp = (np.absolute(hp))**2
                    else              : raise TermError

                    # Compute the empirical interpolation error.
                    hp_interp = np.dot(basis_interpolant,hp[emp_nodes])
                    dh        = hp - hp_interp
                    eies.append(np.real(np.vdot(dh, dh)))
                #FIXME: which one of the errors?
    #                overlap_of_two_waveforms(hp, hp_interp, self.deltaF, self.error_version)
    
                # Select the worst represented point.
                arg_newbasis    = np.argmax(eies)
                new_basis_point = outliers[arg_newbasis]
                
                # Update outliers.
                outliers = outliers[np.array(eies) > training_set_tol]

                # Update the user on how many outliers remain.
                if self.verbose:
                    print("\n{}".format(ndim), "basis elements gave", len(outliers), "outliers with surrogate error >", training_set_tol, " out of {} training points.\n".format(training_set_size))
                    for xy in range(len(outliers)): print("Outlier: {}, with surrogate error {}".format(outliers[xy], eies[np.array(eies) > training_set_tol][xy]))

                # Enrich the basis with the worst outlier.
                if(len(outliers) > 0):
                
                    # Create new basis element.
                    hp_new, _ = self.paramspoint_to_wave(new_basis_point)
                    if   term == 'lin': pass
                    elif term == 'qua': hp_new = (np.absolute(hp_new))**2
                    else              : raise TermError
                    hp_new    = vector_normalised(hp_new)
                    print('FIXME: Sure to gram_schimdt here? Also, normalise after Gram Schmidt?')
                    basis_new = gram_schmidt(known_bases, hp_new)
                    
                    # Append to basis
                    known_bases  = np.append(known_bases,  np.array([basis_new]),       axis=0)
                    known_params = np.append(known_params, np.array([new_basis_point]), axis=0)
                    
                    print('FIXME: track residual modula or sigma_eim better')
                    # residual_modula = np.append(residual_modula, rm_new)

        # Finalise and store the output.
        frequencies = self.freq[emp_nodes]
        np.save(file_interpolant,     basis_interpolant)
        np.save(file_empirical_freqs, frequencies)
        
        return frequencies, basis_interpolant, known_params

    ## Main function starting the ROQ construction.

    def run(self, run_type):
        
        # Initialise data.
        d = {}

        execution_time_presel_basis = time.time()
        # Initialise basis, either using a default choice or a previously constructed one.
        if(self.start_values==None):
            # We choose the first elements of the basis to correspond to the lower and upper values of the parameters range. Note these are not the N-D corners of the parameter space N-cube.
            initial_basis, initial_params, initial_residual_modula = self._construct_corner_basis(run_type)
            # Run a first preselection loop, building a basis of dimension n_pre_basis
            preselection_basis, preselection_params, preselection_residual_modula = self._construct_preselection_basis(initial_basis, initial_params, initial_residual_modula, run_type)
        else:
            # FIXME: load a previously constructed basis.
            preselection_basis, preselection_params, preselection_residual_modula = None, None, None
            raise Exception("Start parameters selected by the user have not yet been implemented.")

        if(self.timing):
            execution_time_presel_basis = (time.time() - execution_time_presel_basis)/60.0
            print('Timing: pre-selection basis with parallel={} [minutes]: {}'.format(self.parallel, execution_time_presel_basis))


        # Internally store the output data for later testing.
        d['{}_pre_bases'.format(run_type)]  = preselection_basis
        d['{}_pre_params'.format(run_type)] = preselection_params
        d['{}_pre_res'.format(run_type)]    = preselection_residual_modula

        # Start the series of loops in which the pre-selected basis is enriched by the outliers found on ever increasing training sets.
        frequencies, basis_interpolant, basis_parameters = self._roqs(preselection_basis, preselection_params, run_type)

        # Internally store the output data for later testing.
        d['{}_interpolant'.format(run_type)] = basis_interpolant
        d['{}_f'.format(run_type)]           = frequencies
        d['{}_emp_nodes'.format(run_type)]   = np.searchsorted(self.freq, d['{}_f'.format(run_type)])
        d['{}_params'.format(run_type)]      = basis_parameters

        return d

if __name__ == '__main__':

    # Initialise and read config.
    parser      = OptionParser(initialise.usage)
    parser.add_option('--config-file', type='string', metavar = 'config_file', default = None)
    (opts,args) = parser.parse_args()
    config_file = opts.config_file

    config_pars, params_ranges, test_values = initialise.read_config(config_file)

    # Point(s) of the parameter space on which to initialise the basis. If not passed by the user, defaults to upper/lower corner of the parameter space.
    start_values = None

    # Initialise ROQ parameters and structures.
    pyroq = PyROQ(config_pars, params_ranges, start_values=start_values)
    freq  = pyroq.freq

    data = {}

    for run_type in config_pars['I/O']['run-types']:

        term = run_type[0:3]
        if not(config_pars['I/O']['post-processing-only']):
            # Create the bases and save ROQ.
            data[run_type] = pyroq.run(term)
        else:
            # Read ROQ from previous run.
            data[run_type]                                = {}
            data[run_type]['{}_f'.format(term)]           = np.load(os.path.join(config_pars['I/O']['output'],'ROQ_data/{type}/empirical_frequencies_{type}.npy'.format(type=run_type)))
            data[run_type]['{}_interpolant'.format(term)] = np.load(os.path.join(config_pars['I/O']['output'],'ROQ_data/{type}/basis_interpolant_{type}.npy'.format(type=run_type)))
            data[run_type]['{}_emp_nodes'.format(term)]   = np.searchsorted(freq, data[run_type]['{}_f'.format(term)])
            data[run_type]['{}_params'.format(term)]      = np.load(os.path.join(config_pars['I/O']['output'],'ROQ_data/{type}/basis_waveform_params_{type}.npy'.format(type=run_type)))

        # Output the basis reduction factor.
        print('\n###########\n# Results #\n###########\n')
        print('{} basis reduction factor: (Original freqs [{}]) / (New freqs [{}]) = {}'.format(run_type, len(freq), len(data[run_type]['{}_f'.format(term)]), len(freq)/len(data[run_type]['{}_f'.format(term)])))

        # Plot the basis parameters corresponding to the selected basis (only the first N elements determined during the interpolant construction procedure).
        post_processing.histogram_basis_params(data[run_type]['{}_params'.format(term)][:len(data[run_type]['{}_f'.format(term)])], pyroq.outputdir, pyroq.i2n)

        # Surrogate tests.
        post_processing.test_roq_error(data[run_type]['{}_interpolant'.format(term)], data[run_type]['{}_emp_nodes'.format(term)], term, pyroq)

        # Plot the representation error for a random waveform, using the interpolant built from the constructed basis. Useful for visual diagnostics.
        print('\n\n##############################################\n# Testing the waveform using the parameters: #\n##############################################\n')
        parampoint_test = []
        print('name    | value | index')
        for name, val in test_values.items():
            print('{} | {}   | {} '.format(name.ljust(len('lambda1')), val, pyroq.n2i[name]))
            parampoint_test.append(val)
        parampoint_test = np.array(parampoint_test)

        post_processing.plot_representation_error(data[run_type]['{}_interpolant'.format(term)], data[run_type]['{}_emp_nodes'.format(term)], parampoint_test, term, pyroq.outputdir, freq, pyroq.paramspoint_to_wave)

    # Show plots, if requested.
    if(config_pars['I/O']['show-plots']): plt.show()
