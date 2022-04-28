# General python imports
import matplotlib, multiprocessing as mp, numpy as np, os, random, warnings
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Package internal import
from wvfwrappers import *

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# PyRQQ
# =====

# Set some defaults
defaults = {}

# Parameter ranges

# This is the training range of MLGW-BNS
defaults['params_ranges'] = {
    'mc'      : [0.9, 1.4]     ,
    'q'       : [1, 3]         ,
    's1s1'    : [0, 0.5]       ,
    's1s2'    : [0, np.pi]     ,
    's1s3'    : [0, 2.0*np.pi] ,
    's2s1'    : [0, 0.5]       ,
    's2s2'    : [0, np.pi]     ,
    's2s3'    : [0, 2.0*np.pi] ,
    'lambda1' : [5, 5000]      ,
    'lambda2' : [5, 5000]      ,
    'iota'    : [0, np.pi]     ,
    'phiref'  : [0, 2*np.pi]   ,
}

# Point of the parameter space on which a targeted check is required
default_test_values = {
        'mc'      : 1.3 ,
        'q'       : 2   ,
        's1s1'    : 0.  ,
        's1s2'    : 0.2 ,
        's1s3'    : -0.1,
        's2s1'    : 0.  ,
        's2s2'    : 0.15,
        's2s3'    : -0.1,
        'lambda1' : 1000,
        'lambda2' : 1000,
        'iota'    : 1.9 ,
        'phiref'  : 0.6 ,
}
# Initialise test_values
test_values = {}

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
                 # Dictionary with any parameter needed for the waveform approximant
                 additional_waveform_params = {},
                 # Intrinsic parameter space on which the interpolants will be constructed
                 params_ranges     = defaults['params_ranges'],
                 # Frequency axis on which the interpolant will be constructed
                 f_min             = 20,
                 f_max             = 1024,
                 deltaF            = 1./4.,
                 
                 # Interpolants construction parameters
                 
                 # Number of random test waveforms. For diagnostics, 1000 is fine. For real ROQs calculation, set it to be 1000000.
                 ntests            = 1000,
                 # Number of points for each search for a new basis element. For diagnostic testing, 30 -100 is fine. For real ROQs computation, this can be 300 to 2000, roughly comparable to the number of basis elments.
                 # What value to choose depends on the nature of the waveform, such as how many features it has. It also depends on the parameter space and the signal length.
                 npts              = 80,
                 
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
        self.approximant       = approximant
        self.additional_waveform_params = additional_waveform_params
        self.params_ranges     = params_ranges
        self.f_min             = f_min
        self.f_max             = f_max
        self.deltaF            = deltaF
        self.distance          = distance

        self.ntests            = ntests
        self.npts              = npts

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
        
        if not os.path.exists(self.outputdir): os.makedirs(self.outputdir)
    
        # Choose waveform
        if self.approximant in WfWrapper.keys():
            self.wvf = WfWrapper[self.approximant](self.approximant, self.additional_waveform_params)
        else:
            raise ValueError('unknown approximant')

        # Build the map between params names and indexes
        self.map_params_indexs() # self.i2n, self.n2i, self.nparams
        
        # Initial basis
        self.freq = np.arange(self.f_min, self.f_max, self.deltaF)
        self.initial_basis() # self.nparams, self.params_low, self.params_hig, self.params_ini, self.hp1
        
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
        Calculating the normalized residual (= a new basis) of a vector vec from known bases
        """
        for i in np.arange(0,len(bases)):
            vec = vec - self.proj(bases[i], vec)
        return vec/np.sqrt(np.vdot(vec,vec)) # normalized new basis
    
    def overlap_of_two_waveforms(self, wf1, wf2):
        """
        Calculating overlap of two waveforms
        """
        wf1norm = wf1/np.sqrt(np.vdot(wf1,wf1)) 
        wf2norm = wf2/np.sqrt(np.vdot(wf2,wf2)) 
        return np.real(np.vdot(wf1norm, wf2norm)) 

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

        # additionally store spin vectors
        if(('s1s1' in self.n2i.keys()) and ('s1s2' in self.n2i.keys()) and ('s1s3' in self.n2i.keys())):
            p['s1sphere'] = p['s1s1'],p['s1s2'],p['s1s3']
        if(('s2s1' in self.n2i.keys()) and ('s2s2' in self.n2i.keys()) and ('s2s3' in self.n2i.keys())):
            p['s2sphere'] = p['s2s1'],p['s2s2'],p['s2s3']
        if(('s1x' in self.n2i.keys()) and ('s1y' in self.n2i.keys()) and ('s1z' in self.n2i.keys())):
            p['s1xyz'] = p['s1x'],p['s1y'],p['s1z']
        if(('s2x' in self.n2i.keys()) and ('s2y' in self.n2i.keys()) and ('s2z' in self.n2i.keys())):
            p['s2xyz'] = p['s2x'],p['s2y'],p['s2z']
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
    
    def _paramspoint_to_wave(self, paramspoint, update_m1m2=True, update_sxyz=True):
        """
        Generate a waveform given a paramspoint
        By default, 
         - it assumes that paramspoint contains (mc,q) and updates (m1,m2) accordingly   
         - if paramspoint contains the spherical spin, then updates the cartesian accordingly
        """
        p = self.update_waveform_params(paramspoint)

        if update_m1m2:
            p['m1'],p['m2'] = self.get_m1m2_from_mcq(p['mc'],p['q'])

        if update_sxyz:
            if(('s1s1' in self.n2i.keys()) and ('s1s2' in self.n2i.keys()) and ('s1s3' in self.n2i.keys())):
                p['s1sphere'] = p['s1s1'],p['s1s2'],p['s1s3']
                p['s1xyz'] = self.spherical_to_cartesian(p['s1sphere']) 
                p['s1x'],p['s1y'],p['s1z'] = p['s1xyz']
                
            if(('s2s1' in self.n2i.keys()) and ('s2s2' in self.n2i.keys()) and ('s2s3' in self.n2i.keys())):
                p['s2sphere'] = p['s2s1'],p['s2s2'],p['s2s3']
                p['s2xyz'] = self.spherical_to_cartesian(p['s2sphere'])
                p['s2x'],p['s2y'],p['s2z'] = p['s2xyz']
        
        hp, _ = self.wvf.generate_waveform(p, self.deltaF, self.f_min, self.f_max, self.distance)
        return hp
    
    def generate_a_waveform_from_mcq(self, paramspoint): 
        """
        This assumes paramspoint contains values for mc,q
        and updates m1,m2 in the waveform parameters before generating
        the waveform
        """
        return self._paramspoint_to_wave(paramspoint)

    def generate_a_waveform(self, paramspoint): 
        """
        This does not assume data for (mc,q)
        """
        return self._paramspoint_to_wave(paramspoint, update_m1m2=False, update_sxyz=True)
    
    def _compute_modulus(self, paramspoint, known_bases, term='lin'):
        
        print(paramspoint)
        exit()
        hp = self._paramspoint_to_wave(paramspoint)
        
        if   term == 'lin' : residual = hp
        elif term == 'quad': residual = (np.absolute(hp))**2
        else               : raise ValueError("unknown term")
        
        if(term=='quad'):
            print(hp, type(hp))
        for k in np.arange(0,len(known_bases)):
            if(term=='quad'):
                print(known_bases[k], type(known_bases[k]))
                exit()
            residual -= self.proj(known_bases[k],hp)
        
        return np.sqrt(np.vdot(residual, residual))
        
    def _least_match_waveform_unnormalized(self, paramspoints, known_bases, term='lin'):
        """
        Now generating N=npts waveforms at points that are 
        randomly uniformly distributed in parameter space
        and calculate their inner products with the 1st waveform
        so as to find the best waveform as the new basis
        """
        if self.parallel:
            paramspointslist = paramspoints.tolist()
            pool = mp.Pool(processes=nprocesses)
            modula = [pool.apply(self._compute_modulus, args=(paramspoint, known_bases, term)) for paramspoint in paramspointslist]
            pool.close()
        else:
            npts   = len(paramspoints) # = self.npts
            modula = np.zeros(npts)
            for i,paramspoint in enumerate(paramspoints):
                modula[i] = np.real(self._compute_modulus(paramspoint, known_bases, term=term))

        arg_newbasis = np.argmax(modula) 
        hp = self._paramspoint_to_wave(paramspoints[arg_newbasis])
        if term == 'lin':
            pass
        elif term == 'quad':
            hp = (np.absolute(hp))**2
        else:
            raise ValueError("unknown term")
        basis_new = self.gram_schmidt(known_bases, hp)
       
        return np.array([basis_new, paramspoints[arg_newbasis], modula[arg_newbasis]]) # elements, masses&spins, residual mod

    def least_match_linear_waveform_unnormalized(self, paramspoints, known_bases):
        return self._least_match_waveform_unnormalized(paramspoints, known_bases, term='lin')

    def least_match_quadratic_waveform_unnormalized(self, paramspoints, known_bases):
        return self._least_match_waveform_unnormalized(paramspoints, known_bases, term='quad')

    def _bases_searching_results_unnormalized(self, known_bases, basis_waveforms, params, residual_modula, term='lin'):
        if term == 'lin':
            nbases = self.nbases
            fbase = self.outputdir+'/linearbases.npy'
            fparams = self.outputdir+'/linearbasiswaveformparams.npy'
        elif term=='quad':
            nbases = self.nbases_quad
            fbase = self.outputdir+'/quadraticbases.npy'
            fparams = self.outputdir+'/quadraticbasiswaveformparams.npy'
        else:
            raise ValueError("unknown term")
    
        print('\n\n###########################\n# Starting {} iteration #\n###########################\n'.format(term.ljust(4)))
        for k in np.arange(0,nbases-1):
            paramspoints = self.generate_params_points()
            basis_new, params_new, rm_new = self._least_match_waveform_unnormalized(paramspoints, known_bases, term=term)
            if self.verbose:
                np.set_printoptions(linewidth=np.inf)
                np.set_printoptions(precision=6)
                np.set_printoptions(suppress=True)
                print("Iter ({}): ".format(term), k+1, " and new basis waveform", params_new)
            known_bases= np.append(known_bases, np.array([basis_new]), axis=0)
            params = np.append(params, np.array([params_new]), axis = 0)
            residual_modula = np.append(residual_modula, rm_new)
        np.save(fbase,known_bases)
        np.save(fparams,params)
        return known_bases, params, residual_modula
    
    def bases_searching_linear_results_unnormalized(self, known_bases, basis_waveforms, params, residual_modula):
        return self._bases_searching_results_unnormalized(known_bases, basis_waveforms, params, residual_modula, term='lin')

    def bases_searching_quadratic_results_unnormalized(self,known_bases, basis_waveforms, params, residual_modula):
        return self._bases_searching_results_unnormalized(known_bases, basis_waveforms, params, residual_modula, term='quad')
    
    def initial_basis(self):
        """
        Initialize parameter ranges and basis
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
#            params_ini_list.append((self.params_low[i] + (self.params_hig[i] - self.params_low[i]) * 0.1)) #CHECKME: this should not be hardcoded
            params_ini_list.append(self.params_low[i]) #CHECKME: this should not be hardcoded

            if self.verbose:
                print('{}    | {} | ( {:.6f} - {:.6f} ) | {:.6f}'.format(str(i).ljust(2),
                                                                      n.ljust(len('lambda1')),
                                                                      self.params_low[i],
                                                                      self.params_hig[i],
                                                                      params_ini_list[i]))
        self.params_ini = np.array([params_ini_list])
        # First waveform
        self.hp1 = self.generate_a_waveform_from_mcq(self.params_low)
        return 

    def empnodes(self, ndim, known_bases, fact=100000000):
        """
        Here known_bases is the full copy known_bases_copy. Its length is equal to or longer than ndim.
        """
        emp_nodes = np.arange(0,ndim) * fact
        emp_nodes[0] = np.argmax(np.absolute(known_bases[0]))
        c1 = known_bases[1,emp_nodes[0]]/known_bases[0,1]
        interp1 = np.multiply(c1,known_bases[0])
        diff1 = interp1 - known_bases[1]
        r1 = np.absolute(diff1)
        emp_nodes[1] = np.argmax(r1)
        for k in np.arange(2,ndim):
            emp_tmp = emp_nodes[0:k]
            Vtmp = np.transpose(known_bases[0:k,emp_tmp])
            inverse_Vtmp = np.linalg.pinv(Vtmp)
            e_to_interp = known_bases[k]
            Ci = np.dot(inverse_Vtmp, e_to_interp[emp_tmp])
            interpolantA = np.zeros(len(known_bases[k]))+np.zeros(len(known_bases[k]))*1j
            for j in np.arange(0, k):
                tmp = np.multiply(Ci[j], known_bases[j])
                interpolantA += tmp
            diff = interpolantA - known_bases[k]
            r = np.absolute(diff)
            emp_nodes[k] = np.argmax(r)
            emp_nodes = sorted(emp_nodes)
        u, c = np.unique(emp_nodes, return_counts=True)
        dup = u[c > 1]
        emp_nodes = np.unique(emp_nodes)
        ndim = len(emp_nodes)
        V = np.transpose(known_bases[0:ndim, emp_nodes])
        inverse_V = np.linalg.pinv(V)
        return np.array([ndim, inverse_V, emp_nodes])

    def empnodes_quad(self, ndim, known_bases):
        return empnodes(self, ndim, known_bases) #CHECKME: this routine appears identical to the above (duplicated in original code?)

    def _surroerror(self, ndim, inverse_V, emp_nodes, known_bases, paramspoint, term = 'lin'):
        hp = self.generate_a_waveform_from_mcq(paramspoint) 
        if term == 'lin':
            pass
        elif term == 'quad':
            hp = (np.absolute(hp))**2
        else:
            raise ValueError("unknown term")
        
        Ci = np.dot(inverse_V, hp[emp_nodes])
        interpolantA = np.zeros(len(hp))+np.zeros(len(hp))*1j
        for j in np.arange(0, ndim):
            tmp = np.multiply(Ci[j], known_bases[j])
            interpolantA += tmp

        return (1-self.overlap_of_two_waveforms(hp, interpolantA))*self.deltaF
    
    def surroerror_lin(self, ndim, inverse_V, emp_nodes, known_bases, paramspoint):
        return self._surroerror(ndim, inverse_V, emp_nodes, known_bases, paramspoint, term = 'lin')

    def surroerror_quad(self, ndim, inverse_V, emp_nodes, known_bases, paramspoint):
        return self._surroerror(ndim, inverse_V, emp_nodes, known_bases, paramspoint, term = 'quad')
    
    def _surros(self, ndim, inverse_V, emp_nodes, known_bases, term='lin'):
        if   term == 'lin':  tol = self.tolerance
        elif term == 'quad': tol = self.tolerance_quad
        else:                raise ValueError("Unknown basis term requested.")
        
        paramspoints = self.generate_params_points(npts=self.ntests)
        surros = np.zeros(self.ntests)
        count = 0
        for i, paramspoint in enumerate(paramspoints):
            surros[i] = self._surroerror(ndim,
                                         inverse_V,
                                         emp_nodes,
                                         known_bases[0:ndim],
                                         paramspoint, 
                                         term = term)
            if (surros[i] > tol):
                count = count+1
        if self.verbose:
            print('\n{}'.format(ndim), "basis elements gave", count, "bad points of surrogate error >", self.tolerance, '\n')
        if count == 0:
            return 0
        else:
            return 1
    
    def surros_lin(self, ndim, inverse_V, emp_nodes, known_bases):
        return self._surros(ndim, inverse_V, emp_nodes, known_bases, term='lin')

    def surros_quad(self, ndim, inverse_V, emp_nodes, known_bases):
        return self._surros(ndim, inverse_V, emp_nodes, known_bases, term='quad')
    
    def _roqs(self, known_bases, term='lin'):
        
        if term == 'lin':
            ndimlow      = self.ndimlow
            ndimhigh     = self.ndimhigh
            ndimstepsize = self.ndimstepsize
            froq         = self.outputdir+'/B_linear.npy'
            fnodes       = self.outputdir+'/fnodes_linear.npy'
        elif term == 'quad':
            ndimlow      = self.ndimlow_quad
            ndimhigh     = self.ndimhigh_quad
            ndimstepsize = self.ndimstepsize_quad
            froq         = self.outputdir+'/B_quadratic.npy'
            fnodes       = self.outputdir+'/fnodes_quadratic.npy'
        else:
              raise ValueError("unknown term")

        for num in np.arange(ndimlow, ndimhigh, ndimstepsize):
            
            ndim, inverse_V, emp_nodes = self.empnodes(num, known_bases)
            
            if self._surros(ndim, inverse_V, emp_nodes, known_bases, term=term) == 0:
                b = np.dot(np.transpose(known_bases[0:ndim]),inverse_V)
                f = self.freq[emp_nodes]
                np.save(froq,np.transpose(b))
                np.save(fnodes,f)
                if self.verbose:
                    print("Number of linear basis elements is", ndim, "and the ROQ data are saved in",froq)
                break

        return b,f

    def roqs_lin(self, known_bases):
        return self._roqs(known_bases, term='lin')

    def roqs_quad(self, known_bases):
        return self._roqs(known_bases, term='quad')

    ## Main function starting the ROQ construction

    def run(self):
        d          = {}
        hp1        = self.hp1
        hp1_quad   = (np.absolute(hp1))**2
        params_ini = self.params_ini
        
        # Search for linear basis elements to build & save linear ROQ data in the local directory.
        known_bases_start     = np.array([hp1/np.sqrt(np.vdot(hp1,hp1))])
        basis_waveforms_start = np.array([hp1])
        residual_modula_start = np.array([0.0])
        
        bases, params, residual_modula = self._bases_searching_results_unnormalized(known_bases_start, basis_waveforms_start, params_ini, residual_modula_start, term='lin')
        B, f = self._roqs(bases, term='lin')

        d['lin_B']      = B
        d['lin_f']      = f
        d['lin_bases']  = bases
        d['lin_params'] = params
        d['lin_res']    = residual_modula
        
        # Search for quadratic basis elements to build & save quadratic ROQ data.
        known_bases_start     = np.array([hp1_quad/np.sqrt(np.vdot(hp1_quad,hp1_quad))])
        basis_waveforms_start = np.array([hp1_quad])
        residual_modula_start = np.array([0.0])
        
        bases, params, residual_modula = self._bases_searching_results_unnormalized(known_bases_start, basis_waveforms_start, params_ini, residual_modula_start, term='quad')
        B, f = self._roqs(bases, term='quad')
        
        d['quad_B']      = B
        d['quad_f']      = f
        d['quad_bases']  = bases
        d['quad_params'] = params
        d['quad_res']    = residual_modula
        
        return d
    
    ## Functions to test the performance of the ROQ
    
    def _testrep(self, b, emp_nodes, paramspoint, term='lin', show=True):
        hp = self.generate_a_waveform_from_mcq(paramspoint) 
        if term == 'lin':
            pass
        elif term == 'quad':
            hp = (np.absolute(hp))**2
        else:
            raise ValueError("unknown term") 
        hp_emp = hp[emp_nodes]
        hp_rep = np.dot(b,hp_emp)
        diff = hp_rep - hp
        rep_error = diff/np.sqrt(np.vdot(hp,hp))
        freq = self.freq
        if show:
            
            plt.figure(figsize=(15,9))
            plt.plot(freq, np.real(hp), label='Real part of h+ (full)')
            plt.plot(freq, np.real(hp_rep), label='Real part of h+ (ROQ)')
            plt.xlabel('Frequency')
            plt.ylabel('Waveform')
            plt.title('Waveform comparison')
            plt.legend(loc=0)
          
            plt.figure(figsize=(15,9))
            plt.plot(freq, np.imag(hp), label='Imag part of h+ (full)')
            plt.plot(freq, np.imag(hp_rep), label='Imag part of h+ (ROQ)')
            plt.xlabel('Frequency')
            plt.ylabel('Waveform')
            plt.title('Waveform comparison')
            plt.legend(loc=0)
          
            plt.figure(figsize=(15,9))
            plt.plot(freq, np.real(rep_error), label='Real part of h+') 
            plt.plot(freq, np.imag(rep_error), label='Imaginary part of h+')
            plt.xlabel('Frequency')
            plt.ylabel('Fractional Representation Error')
            plt.title('Representation Error')
            plt.legend(loc=0)
            plt.show()
        return freq, rep_error
    
    def testrep_lin(self, b, emp_nodes, paramspoint, show=True):
        return self._testrep(b, emp_nodes, paramspoint, term='lin', show=show)

    def testrep_quad(self, b, emp_nodes, paramspoint, show=True):
        return self._testrep(b, emp_nodes, paramspoint, term='quad', show=show)
    
    def surros_of_test_samples(self, b_linear, emp_nodes, nsamples=0, term='lin'):
        
        if   term == 'lin':  tol = self.tolerance
        elif term == 'quad': tol = self.tolerance_quad
        else:                raise ValueError("Unknown basis term requested.")
        
        if nsamples <= 0: nsamples = self.ntests
        ndim = len(emp_nodes)
        paramspoints = self.generate_params_points(npts=nsamples)
        surros = np.zeros(self.ntests)
        for i,paramspoint in enumerate(paramspoints):
            hp = self.generate_a_waveform_from_mcq(paramspoint)
            hp_emp = hp[emp_nodes]
            hp_rep = np.dot(b_linear,hp_emp) 
            surros[i] = (1-self.overlap_of_two_waveforms(hp, hp_rep))*self.deltaF
            if self.verbose:
                if (surros[i] > tol):
                    print("iter", i, surros[i], points[i])
                if i%100==0:
                    print("iter", i, surros[i])
        return surros


if __name__ == '__main__':

    approx = lalsimulation.IMRPhenomPv2 # 'teobresums-giotto-FD' #'mlgw-bns'

    # Range on which to train the ROQ
    params_ranges = {
        'mc'      : [30, 31]       ,
        'q'       : [1, 1.2]       ,
        's1s1'    : [0, 0.2]       ,
        's1s2'    : [0, np.pi]     ,
        's1s3'    : [0, 2.0*np.pi] ,
        's2s1'    : [0, 0.2]       ,
        's2s2'    : [0, np.pi]     ,
        's2s3'    : [0, 2.0*np.pi] ,
        'lambda1' : [0, 0]         ,
        'lambda2' : [0, 0]         ,
        'iota'    : [0, np.pi]     ,
        'phiref'  : [0, 2*np.pi]   ,
    }

    # Point of the parameter space on which a targeted check is required
    test_values = {
        'mc'      : 30.5 ,
        'q'       : 1.1 ,
        's1s1'    : 0.1 ,
        's1s2'    : 0.2 ,
        's1s3'    : 0.1,
        's2s1'    : 0.05 ,
        's2s2'    : 0.15,
        's2s3'    : 0.1,
        'lambda1' : 0,
        'lambda2' : 0,
        'iota'    : 1.9 ,
        'phiref'  : 0.6 ,
    }

    print('CHECKME: spins in spherical or cartesian?')

    # Initialise ROQ
    pyroq = PyROQ(approximant       = approx,
                  params_ranges     = params_ranges,
                  
                  f_min             = 50,
                  f_max             = 1024,
                  deltaF            = 1./1.,
                  
                  ntests            = 123,
                  npts              = 80,
                  
                  nbases            = 30,
                  ndimlow           = 20,
                  ndimstepsize      = 1,
                  tolerance         = 1e-4,
                  
                  nbases_quad       = 30,
                  ndimlow_quad      = 20,
                  ndimstepsize_quad = 1,
                  tolerance_quad    = 1e-5,
                  
                  parallel          = False,
                  nprocesses        = 4,
                  
                  outputdir         ='./test',
                  verbose           = True,
                  )

    # Create the bases and save them
    data = pyroq.run()

    freq = pyroq.freq
    data['lin_emp_nodes']  = np.searchsorted(freq, data['lin_f'])
    data['quad_emp_nodes'] = np.searchsorted(freq, data['quad_f'])

    # Test waveform

    print('Linear basis reduction factor: (Original freqs [{}]) / (New freqs [{}]) = {}'.format(len(freq), len(data['lin_f']), len(freq)/len(data['lin_f'])))

    print('Quadratic basis reduction factor: (Original freqs [{}]) / (New freqs [{}]) = {}'.format(len(freq), len(data['quad_f']), len(freq)/len(data['quad_f'])))


    if not(test_values): test_values = default_test_values
    parampoint = []
    for name, val in test_values.items():
        print(name, val, pyroq.n2i[name])
        parampoint.append(val)

    parampoint = np.array(parampoint)
    pyroq.testrep_lin( data['lin_B'] , data['lin_emp_nodes'] , parampoint)
    pyroq.testrep_quad(data['quad_B'], data['quad_emp_nodes'], parampoint)

    surros = pyroq.surros_of_test_samples(data['lin_B'], data['lin_emp_nodes'], term='lin')

    plt.figure(figsize=(15,9))
    plt.semilogy(surros,'o',color='black')
    plt.xlabel("Number of Random Test Points")
    plt.ylabel("Surrogate Error")
    plt.title(approx)
    plt.savefig("SurrogateErrorsRandomTestPoints.png")
    plt.show()

#    os.system('mv ./testrep.png ./testrepquad.png {}/.'.format(run_tag))
