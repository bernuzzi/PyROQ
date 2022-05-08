# General python imports
import matplotlib, matplotlib.pyplot as plt, multiprocessing as mp, numpy as np, os, random, warnings

# Package internal import
from wvfwrappers import *

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
np.set_printoptions(linewidth=np.inf)
TermError    = ValueError("Unknown basis term requested.")
VersionError = ValueError("Unknown version requested.")
np.random.seed(150914)

# PyRQQ
# =====

# Set some defaults
defaults = {}

# Parameter ranges

# This is the training range of MLGW-BNS
defaults['params_ranges'] = {
    'mc'      : [0.9, 1.4]   ,
    'q'       : [1, 3]       ,
    's1x'     : [0, 0]       ,
    's1y'     : [0, 0]       ,
    's1z'     : [-0.5, 0.5]  ,
    's2x'     : [0, 0]       ,
    's2y'     : [0, 0]       ,
    's2z'     : [-0.5, 0.5]  ,
    'lambda1' : [5, 5000]    ,
    'lambda2' : [5, 5000]    ,
    'iota'    : [0, np.pi]   ,
    'phiref'  : [0, 2*np.pi] ,
}

defaults['start_values'] = {
    'mc'      : defaults['params_ranges']['mc'][0]     ,
    'q'       : defaults['params_ranges']['q'][0]      ,
    's1x'     : defaults['params_ranges']['s1x'][0]    ,
    's1y'     : defaults['params_ranges']['s1y'][0]    ,
    's1z'     : defaults['params_ranges']['s1z'][0]    ,
    's2x'     : defaults['params_ranges']['s2x'][0]    ,
    's2y'     : defaults['params_ranges']['s2y'][0]    ,
    's2z'     : defaults['params_ranges']['s2z'][0]    ,
    'lambda1' : defaults['params_ranges']['lambda1'][0],
    'lambda2' : defaults['params_ranges']['lambda2'][0],
    'iota'    : defaults['params_ranges']['iota'][0]   ,
    'phiref'  : defaults['params_ranges']['phiref'][0] ,
    }

# Point of the parameter space on which a targeted check is required
defaults['test_values'] = {
        'mc'      : 1.3 ,
        'q'       : 2   ,
        's1x'     : 0.  ,
        's1y'     : 0   ,
        's1z'     : 0.2 ,
        's2x'     : 0   ,
        's2y'     : 0   ,
        's2z'     : 0.1 ,
        'lambda1' : 1000,
        'lambda2' : 1000,
        'iota'    : 1.9 ,
        'phiref'  : 0.6 ,
}

class PyROQ:
    """
    PyROQ Class
    
    * Works with a list of very basic waveform wrappers provided in wvfwrappers.py

    * Keywords for parameters:

    'mc'       : chirp mass
    'm1'       : mass object 1 [Mo]
    'm2'       : mass object 2 [Mo]
    'q'        : mass ratio
    's1s[123]' : spin components object 1, spherical coords (3)
    's2s[123]' : spin components object 2, "         "
    's1[xyz]'  : spin components object 1, cartesian coords (3)
    's2[xyz]'  : spin components object 2, "         "
    'ecc'      : eccentricity
    'lambda1'  : tidal polarizability parameter object 1
    'lambda2'  : tidal polarizability parameter object 2
    'iota'     : inclination
    'phiref'   : reference phase
    'distance' : distance [Mpc]

    Waveform wrappers must work with these keywords

    * The parameter space is *defined* by the keywords of 'params_ranges' 

    """
    def __init__(self,
                 approximant       = 'teobresums-giotto',
                 mc_q_par          = True,
                 spin_sph          = False,

                 # Dictionary with any parameter needed for the waveform approximant
                 additional_waveform_params = {},
                 # Intrinsic parameter space on which the interpolants will be constructed
                 params_ranges     = defaults['params_ranges'],
                 start_values      = defaults['start_values'],

                 # Frequency axis on which the interpolant will be constructed
                 f_min             = 20,
                 f_max             = 1024,
                 deltaF            = 1./4.,
                 
                 # Interpolants construction parameters
                 
                 # Number of random test waveforms. This is the number of cases that are checked to be below tolerance before stopping the ROQ construction.
                 # For diagnostics, 1000 is fine. For real ROQs calculation, set it to be 1000000.
                 ntests            = 1000,
                 # Number of points for each search of a new basis element. For diagnostic testing, 30 -100 is fine. For real ROQs computation, this can be 300 to 2000, roughly comparable to the number of basis elements.
                 # What value to choose depends on the nature of the waveform, such as how many features it has. It also depends on the parameter space and the signal length.
                 npts              = 80,
                 error_version     = 'v1',
                 
                 # Specify the number of linear basis elements. Put your estimation here for the chunk of parameter space.
                 nbases            = 80,
                 # Your estimation of fewest basis elements needed for this chunk of parameter space.
                 ndimlow           = 40,
                 # Number of linear basis elements increament to check if the basis satisfies the tolerance.
                 ndimstepsize      = 10,
                 # Surrogage error threshold for linear basis elements
                 tolerance         = 1e-8,
                 
                 # Specify the number of quadratic basis elements, depending on the tolerance_quad, usually two thirds of that for linear basis
                 nbases_quad       = 80,
                 ndimlow_quad      = 20,
                 ndimstepsize_quad = 10,
                 # Surrogage error threshold for quadratic basis elements
                 tolerance_quad    = 1e-10,
                 
                 # Computing parameters
                 parallel          = False, # The parallel=True will turn on multiprocesses to search for a new basis. To turn it off, set it to be False. Do not turn it on if the waveform generation is not slow compared to data reading and writing to files. This is more useful when each waveform takes larger than 0.01 sec to generate.
                 # Set the number of parallel processes when searching for a new basis.  nprocesses=mp.cpu_count()
                 nprocesses        = 4,
                 
                 outputdir         = './',
                 verbose           = True,
                 
                 # Dummy value, distance does not enter the interpolants construction
                 distance          = 10,  # 10 Mpc is default
                 ):

        # Read input params
        self.approximant                = approximant
        self.mc_q_par                   = mc_q_par
        self.spin_sph                   = spin_sph
        self.params_ranges              = params_ranges
        self.start_values               = start_values
        self.additional_waveform_params = additional_waveform_params

        self.f_min             = f_min
        self.f_max             = f_max
        self.deltaF            = deltaF
        self.distance          = distance

        self.ntests            = ntests
        self.npts              = npts
        self.error_version     = error_version

        # linear basis
        self.nbases            = nbases
        self.ndimlow           = ndimlow
        self.ndimhigh          = nbases+1
        self.ndimstepsize      = ndimstepsize
        self.tolerance         = tolerance

        # quadratic basis
        self.nbases_quad       = nbases_quad
        self.ndimlow_quad      = ndimlow_quad
        self.ndimhigh_quad     = nbases_quad+1
        self.ndimstepsize_quad = ndimstepsize_quad
        self.tolerance_quad    = tolerance_quad

        self.parallel          = parallel
        self.nprocesses        = nprocesses
        
        self.outputdir         = outputdir
        self.verbose           = verbose
        
        # Sanity checks
        if(self.mc_q_par and (('m1' in self.params_ranges) or ('m2' in self.params_ranges))):
            raise ValueError("Cannot pass 'm1' or 'm2' in params_ranges with the 'mc_q_par' option activated.")
        elif(not(self.mc_q_par) and (not('m1' in self.params_ranges) or not('m2' in self.params_ranges))):
            raise ValueError("Need to pass 'm1' and 'm2' in params_ranges with the 'mc_q_par' option de-activated.")

        if(self.spin_sph and (('s1x' in self.params_ranges) or ('s1y' in self.params_ranges) or ('s1z' in self.params_ranges) or ('s2x' in self.params_ranges) or ('s2y' in self.params_ranges) or ('s2z' in self.params_ranges))):
            raise ValueError("Cannot pass 's1[xyz]' or 's2[xyz]' in params_ranges with the 'spin_sph' option activated.")
        elif(not(self.spin_sph) and (not('s1x' in self.params_ranges) or not('s1y' in self.params_ranges) or not('s1z' in self.params_ranges) or not('s2x' in self.params_ranges) or not('s2y' in self.params_ranges) or not('s2z' in self.params_ranges))):
            raise ValueError("Need to pass 's1[x,y,z]' and 's2[x,y,z]' in params_ranges with the 'spin_sph' option de-activated.")

        if(not(self.ndimlow>1) or not(self.ndimlow_quad>1)): raise ValueError("The minimum number of basis elements has to be larger than 1.")
        
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
        if npts <= 0: npts = self.npts
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
            pool = mp.Pool(processes=nprocesses)
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
       
        return np.array([basis_new, paramspoints[arg_newbasis], modula[arg_newbasis]]) # elements, masses&spins, residual mod
            
    def _bases_searching_results_unnormalized(self, known_bases, basis_waveforms, params, residual_modula, term):
        
        if term == 'lin':
            nbases  = self.nbases
            fbase   = self.outputdir+'/ROQ_data/Linear/linearbases.npy'
            fparams = self.outputdir+'/ROQ_data/Linear/linearbasiswaveformparams.npy'
        elif term=='quad':
            nbases  = self.nbases_quad
            fbase   = self.outputdir+'/ROQ_data/Quadratic/quadraticbases.npy'
            fparams = self.outputdir+'/ROQ_data/Quadratic/quadraticbasiswaveformparams.npy'
        else:
            raise TermError
    
        # This block generates a basis of dimension nbases (maximum dimension selected by the user).
        print('\n\n###########################\n# Starting {} iteration #\n###########################\n'.format(term.ljust(4)))
        for k in np.arange(0,nbases-1):
            
            # Generate npts random waveforms.
            paramspoints = self.generate_params_points()
            
            # From the npts randomly generated waveforms, select the worst represented one (i.e. with the largest residuals after basis projection).
            basis_new, params_new, rm_new = self._least_match_waveform_unnormalized(paramspoints, known_bases, term)
            if self.verbose:
                np.set_printoptions(suppress=True)
                print("Iter: ".format(term), k+1, " -- New basis waveform:", params_new)
                np.set_printoptions(suppress=False)

            # The worst represented waveform becomes the new basis element.
            known_bases     = np.append(known_bases, np.array([basis_new]), axis=0)
            params          = np.append(params, np.array([params_new]), axis = 0)
            residual_modula = np.append(residual_modula, rm_new)

        # Store the constructed basis.
        np.save(fbase,known_bases)
        np.save(fparams,params)

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

    def empnodes(self, ndim, known_bases, fact=100000000):
        
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

    def _surroerror(self, ndim, inverse_V, emp_nodes, known_bases, paramspoint, term):
        
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
    
    def _surros(self, ndim, inverse_V, emp_nodes, known_bases, term):
        
        """
            Basis construction stopping function.
            Compute the overlap representation error on a set of random points, using the given basis. If all of them are below tolerance, this is the basis we are searching for.
        """
        
        # Initialise tolerance, parameter space and data structures.
        if   term == 'lin':  tol = self.tolerance
        elif term == 'quad': tol = self.tolerance_quad
        else:                raise TermError
        paramspoints = self.generate_params_points(npts=self.ntests)
        surros       = np.zeros(self.ntests)
        count        = 0
        
        # Compute the overlap representation error
        for i, paramspoint in enumerate(paramspoints):
            surros[i] = self._surroerror(ndim,
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
            ndimlow      = self.ndimlow
            ndimhigh     = self.ndimhigh
            ndimstepsize = self.ndimstepsize
            froq         = self.outputdir+'/ROQ_data/Linear/B_linear.npy'
            fnodes       = self.outputdir+'/ROQ_data/Linear/fnodes_linear.npy'
        elif term == 'quad':
            ndimlow      = self.ndimlow_quad
            ndimhigh     = self.ndimhigh_quad
            ndimstepsize = self.ndimstepsize_quad
            froq         = self.outputdir+'/ROQ_data/Quadratic/B_quadratic.npy'
            fnodes       = self.outputdir+'/ROQ_data/Quadratic/fnodes_quadratic.npy'
        else:
              raise TermError

        # Start from a user-selected minimum number of basis elements and keep adding elements until that basis represents well enough a sufficient number of random waveforms or until you hit the user-selected maximum basis elements number.
        flag = 0
        for num in np.arange(ndimlow, ndimhigh, ndimstepsize):
            
            # Build the empirical interpolation nodes for this basis.
            ndim, inverse_V, emp_nodes = self.empnodes(num, known_bases)
            
            # If the overlap representation error is below tolerance, stop the iteration and store this basis.
            if(self._surros(ndim, inverse_V, emp_nodes, known_bases, term) == 0):
                
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
    
    ## Functions to test the performance of the waveform representation using the interpolant built from the basis.
    
    def testrep(self, b, emp_nodes, paramspoint, term):
        
        hp, _ = self._paramspoint_to_wave(paramspoint)
        
        if   term == 'lin' : pass
        elif term == 'quad': hp = (np.absolute(hp))**2
        else               : raise TermError
        
        hp_emp    = hp[emp_nodes]
        hp_rep    = np.dot(b,hp_emp)
        diff      = hp_rep - hp
        rep_error = diff/np.sqrt(np.vdot(hp,hp))
        freq      = self.freq
            
        plt.figure(figsize=(8,5))
        plt.plot(freq, np.real(hp),     label='Real part of h+ (full)')
        plt.plot(freq, np.real(hp_rep), label='Real part of h+ (ROQ)')
        plt.xlabel('Frequency')
        plt.ylabel('Waveform')
        plt.title('Waveform comparison ({})'.format(term.ljust(4)))
        plt.savefig(os.path.join(self.outputdir,'Plots/Waveform_comparison_real_{}.png'.format(term)))
        plt.legend(loc=0)
      
        plt.figure(figsize=(8,5))
        plt.plot(freq, np.imag(hp),     label='Imag part of h+ (full)')
        plt.plot(freq, np.imag(hp_rep), label='Imag part of h+ (ROQ)')
        plt.xlabel('Frequency')
        plt.ylabel('Waveform')
        plt.title('Waveform comparison ({})'.format(term.ljust(4)))
        plt.savefig(os.path.join(self.outputdir,'Plots/Waveform_comparison_imag_{}.png'.format(term)))
        plt.legend(loc=0)
        
        plt.figure(figsize=(8,5))
        plt.plot(freq, np.real(rep_error), label='Real      part of h+')
        plt.plot(freq, np.imag(rep_error), label='Imaginary part of h+')
        plt.xlabel('Frequency')
        plt.ylabel('Fractional Representation Error')
        plt.title('Representation Error ({})'.format(term.ljust(4)))
        plt.savefig(os.path.join(self.outputdir,'Plots/Representation_error_{}.png'.format(term)))
        plt.legend(loc=0)
        
        return freq, rep_error
    
    def surros_of_test_samples(self, b_linear, emp_nodes, term, nsamples=0):
        
        if   term == 'lin':  tol = self.tolerance
        elif term == 'quad': tol = self.tolerance_quad
        else:                raise TermError
        
        if nsamples <= 0: nsamples = self.ntests
        ndim         = len(emp_nodes)
        surros       = np.zeros(self.ntests)
        paramspoints = self.generate_params_points(npts=nsamples)
        
        print('\n\n###########################################\n# Starting surrogate tests {} iteration #\n###########################################\n'.format(term.ljust(4)))
        for i,paramspoint in enumerate(paramspoints):
            
            hp, _     = self._paramspoint_to_wave(paramspoint)
            hp_emp    = hp[emp_nodes]
            hp_rep    = np.dot(b_linear,hp_emp)
            
            surros[i] = self.overlap_of_two_waveforms(hp, hp_rep)

            np.set_printoptions(suppress=True)
            if self.verbose:
                if (surros[i] > tol):
                    print("Above tolerance (tol={}): Iter: ".format(tol), i, "Surrogate value: ", surros[i], "Parameters: ", paramspoints[i])
                if i%100==0:
                    print("Rolling check (every 100 steps): Iter: ",             i, "Surrogate value: ", surros[i])
            np.set_printoptions(suppress=False)
    
        plt.figure(figsize=(8,5))
        plt.semilogy(surros,'o',color='black')
        plt.xlabel("Number of Random Test Points")
        plt.ylabel("Surrogate Error ({})".format(term.ljust(4)))
        plt.savefig(os.path.join(self.outputdir,"Plots/Surrogate_errors_random_test_points_{}.png".format(term)))
    
        return surros


if __name__ == '__main__':

    show               = False
    test_and_plot_only = False
    mc_q_par           = True  # If true, mass ranges have to be passed through mc and q
    spin_sph           = False # If true, spin ranges have to be passed in spherical coordinates (see PyROQ class description for the names)
    approx             = lalsimulation.IMRPhenomPv2 # 'teobresums-giotto' #'mlgw-bns'
    error_version      = 'v1'
    output             = './test'
    
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

    # Range on which to train the ROQ
    params_ranges = {
        'mc'      : [30, 31]    ,
        'q'       : [1, 1.2]    ,
        's1x'     : [0, 0]      ,
        's1y'     : [0, 0]      ,
        's1z'     : [0.0, 0.2] ,
        's2x'     : [0, 0]      ,
        's2y'     : [0, 0]      ,
        's2z'     : [0.0, 0.2] ,
        'lambda1' : [0, 0]      ,
        'lambda2' : [0, 0]      ,
        'iota'    : [0, np.pi]  ,
        'phiref'  : [0, 2*np.pi],
    }

    start_values = {
        'mc'      : params_ranges['mc'][0]     ,
        'q'       : params_ranges['q'][0]      ,
        's1x'     : params_ranges['s1x'][0]    ,
        's1y'     : params_ranges['s1y'][0]    ,
        's1z'     : params_ranges['s1z'][0]    ,
        's2x'     : params_ranges['s2x'][0]    ,
        's2y'     : params_ranges['s2y'][0]    ,
        's2z'     : params_ranges['s2z'][0]    ,
        'lambda1' : params_ranges['lambda1'][0],
        'lambda2' : params_ranges['lambda2'][0],
        'iota'    : params_ranges['iota'][0]   ,
        'phiref'  : params_ranges['phiref'][0] ,
        }

    # Point of the parameter space on which a targeted check is required
    test_values = {
        'mc'      : 30.5,
        'q'       : 1.1 ,
        's1x'     : 0.  ,
        's1y'     : 0   ,
        's1z'     : 0.2 ,
        's2x'     : 0   ,
        's2y'     : 0   ,
        's2z'     : 0.1 ,
        'lambda1' : 0   ,
        'lambda2' : 0   ,
        'iota'    : 1.9 ,
        'phiref'  : 0.6 ,
    }

    # Point(s) of the parameter space on which to initialise the basis. If not passed by the user, select defaults.
    if not('params_ranges' in locals() or 'params_ranges' in globals()): params_ranges = defaults['params_ranges']
    if not('start_values'  in locals() or 'start_values'  in globals()): start_values  = defaults['start_values']

    # Initialise ROQ
    pyroq = PyROQ(approximant       = approx,
                  mc_q_par          = mc_q_par,
                  spin_sph          = spin_sph,
                  params_ranges     = params_ranges,
                  start_values      = start_values,
                  
                  f_min             = 50,
                  f_max             = 1024,
                  deltaF            = 1./1.,
                  
                  ntests            = 6,
                  npts              = 80,
                  error_version     = error_version,
                  
                  nbases            = 20,
                  ndimlow           = 10,
                  ndimstepsize      = 1,
                  tolerance         = 1e-4,
                  
                  nbases_quad       = 20,
                  ndimlow_quad      = 10,
                  ndimstepsize_quad = 1,
                  tolerance_quad    = 1e-5,
                  
                  parallel          = False,
                  nprocesses        = 4,
                  
                  outputdir         = output,
                  verbose           = True,
                  )


    ##############################################
    # No parameter should be changed below here. #
    ##############################################

    freq = pyroq.freq

    if not(test_and_plot_only):
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
    if not('test_values' in locals() or 'test_values' in globals()): test_values = defaults['test_values']
    parampoint = []
    print('name    | value | index')
    for name, val in test_values.items():
        print('{} | {}   | {} '.format(name.ljust(len('lambda1')), val, pyroq.n2i[name]))
        parampoint.append(val)
    parampoint = np.array(parampoint)

    # Now est the performance of the representation of a random waveform, using the interpolant built from the constructed basis.
    pyroq.testrep(data['lin_B'] , data['lin_emp_nodes'] , parampoint, 'lin')
    pyroq.testrep(data['quad_B'], data['quad_emp_nodes'], parampoint, 'quad')

    # Surrogate tests
    surros = pyroq.surros_of_test_samples(data['lin_B'] , data['lin_emp_nodes'] , 'lin')
    surros = pyroq.surros_of_test_samples(data['quad_B'], data['quad_emp_nodes'], 'quad')

    if(show): plt.show()
