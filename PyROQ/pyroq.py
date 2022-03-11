import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
#import h5py
import warnings
warnings.filterwarnings('ignore')
import random
import multiprocessing as mp
#from mpl_toolkits.mplot3d import axes3d

from wvfwrappers import *

# PyRQQ
# =====

defaults = {}
defaults['intrinsic_params_defaults'] = {
    'mc'      : [20, 30],
    'q'       : [1, 2],
    's1sphere': [[0, 0, 0], [0.2, np.pi, 2.0*np.pi]],
    's2sphere': [[0, 0, 0], [0.2, np.pi, 2.0*np.pi]],
    'ecc'     : [0.0, 0.2],
    'lambda1' : [0, 1000],
    'lambda2' : [0, 1000],
    'iota'    : [0, np.pi],
    'phiref' : [0, 2*np.pi],
}

class PyROQ:
    def __init__(self,
                 approximant = 'teobresums-giotto-FD',
                 # Intrinsic parameter space on which the interpolants will be constructed
                 intrinsic_params = defaults['intrinsic_params_defaults'],
                 # Frequency axis on which the interpolant will be constructed
                 f_min = 20,
                 f_max = 1024,
                 deltaF = 1./4.,
                 # Dummy value, distance does not enter the interolants construction
                 distance = 10 * LAL_PC_SI * 1.0e6,  # 10 Mpc is default 
                 # Computing parameters
                 parallel = False, # The parallel=True will turn on multiprocesses to search for a new basis. To turn it off, set it to be False.
                 # Do not turn it on if the waveform generation is not slow compared to data reading and writing to files.
                 # This is more useful when each waveform takes larger than 0.01 sec to generate.
                 # Set the number of parallel processes when searching for a new basis.  nprocesses=mp.cpu_count()
                 nprocesses = 4,
                 # Interpolants construction parameters
                 nts = 123, # Number of random test waveforms
                 # For diagnostics, 1000 is fine.
                 # For real ROQs calculation, set it to be 1000000.
                 npts = 80, # Specify the number of points for each search for a new basis element
                 # For diagnostic testing, 30 -100 is fine. 
                 # For real ROQs computation, this can be 300 to 2000, roughly comparable to the number of basis elments.
                 # What value to choose depends on the nature of the waveform, such as how many features it has. 
                 # It also depends on the parameter space and the signal length. 
                 # Specify the number of linear basis elements. Put your estimation here for the chunk of parameter space.
                 nbases = 80,
                 # Your estimation of fewest basis elements needed for this chunk of parameter space.
                 ndimlow = 40,
                 # Number of linear basis elements increament to check if the basis satisfies the tolerance.
                 ndimstepsize = 10,
                 # Surrogage error threshold for linear basis elements
                 tolerance = 1e-8,
                 # Specify the number of quadratic basis elements, depending on the tolerance_quad, usually two thirds of that for linear basis
                 nbases_quad = 80,
                 ndimlow_quad = 20,
                 # Surrogage error threshold for quadratic basis elements
                 tolerance_quad = 1e-10,
                 outputdir = './',
                 verbose = True,
                 ):

        self.approximant = approximant
        self.intrinsic_params = intrinsic_params
        self.f_min = f_min
        self.f_max = f_max
        self.deltaF = deltaF 
        self.distance = distance
        self.parallel = parallel
        self.nprocesses = nprocesses
        self.nts = nts 
        self.npts = npts 
        self.nbases = nbases
        self.ndimlow = ndimlow
        self.tolerance = tolerance
        self.nbases_quad  = nbases_quad 
        self.ndimlow_quad = ndimlow_quad
        self.ndimstepsize_quad = ndimstepsize_quad
        self.tolerance_quad = tolerance_quad

        self.outputdir = outputdir
        self.verbose = verbose

        self.ndimhigh = nbases+1 
        self.ndimhigh_quad = nbases_quad+1
        
        # Choose waveform
        if self.approximant in WfWrapper.keys():
            self.wvf = WfWrapper[self.approximant]
        else:
            raise ValueError('unknown approximant')

        # Initial basis
        self.freq = np.arange(f_min,f_max,deltaF)
        self.initial_basis() # self.nparams, self.params_low, self.params_high, self.params_start, self.hp1
        
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
            vec = vec - proj(bases[i], vec)
        return vec/np.sqrt(np.vdot(vec,vec)) # normalized new basis
    
    def overlap_of_two_waveforms(self, wf1, wf2):
        """
        Calculating overlap of two waveforms
        """
        wf1norm = wf1/np.sqrt(np.vdot(wf1,wf1)) # normalize the first waveform
        wf2norm = wf2/np.sqrt(np.vdot(wf2,wf2)) # normalize the second waveform
        diff = wf1norm - wf2norm
        #overlap = 1 - 0.5*(np.vdot(diff,diff))
        overlap = np.real(np.vdot(wf1norm, wf2norm))
        return overlap

    def spherical_to_cartesian(self, sph):
        x = sph[0]*np.sin(sph[1])*np.cos(sph[2])
        y = sph[0]*np.sin(sph[1])*np.sin(sph[2])
        z = sph[0]*np.cos(sph[1])
        car = [x,y,z]
        return car

    def get_m1m2_from_mcq(self, mc,q):
        m2 = mc * q ** (-0.6) * (1+q)**0.2
        m1 = m2 * q
        return np.array([m1,m2])

    def generate_a_waveform(self, m1, m2, spin1, spin2, ecc, lambda1, lambda2, iota, phiRef):
        test_mass1 = m1 * LAL_MSUN_SI
        test_mass2 = m2 * LAL_MSUN_SI
        hp, hc = self.wvf.generate_waveform(test_mass1, test_mass2, spin1, spin2, ecc, lambda1, lambda2, iota, phiRef,
                                            self.distance, self.deltaF, self.f_min, self.f_max)
        return hp
    
    def generate_a_waveform_from_mcq(self, mc, q, spin1, spin2, ecc, lambda1, lambda2, iota, phiRef):
        m1,m2 = self.get_m1m2_from_mcq(mc,q)
        test_mass1 = m1 * LAL_MSUN_SI
        test_mass2 = m2 * LAL_MSUN_SI
        hp, hc = self.wvf.generate_waveform(test_mass1, test_mass2, spin1, spin2, ecc, lambda1, lambda2, iota, phiRef,
                                            self.distance, self.deltaF, self.f_min, self.f_max)
        return hp

    def generate_params_points(self):
        paramspoints = np.random.uniform(self.params_low, self.params_high, size=(self.npts,self.nparams))
        paramspoints = paramspoints.round(decimals=6)
        return paramspoints

    def compute_modulus(self, paramspoint, known_bases):
        m1, m2 = self.get_m1m2_from_mcq(paramspoint[0],paramspoint[1])
        s1x, s1y, s1z = self.spherical_to_cartesian(paramspoint[2:5]) 
        s2x, s2y, s2z = self.spherical_to_cartesian(paramspoint[5:8]) 
        iota = paramspoint[8]  
        phiRef = paramspoint[9]
        ecc = 0
        if len(paramspoint)==11:
            ecc = paramspoint[10]
        if len(paramspoint)==12:
            lambda1 = paramspoints[arg_newbasis][10]
            lambda2 = paramspoints[arg_newbasis][11]
        f_ref = 0 
        RA = 0    
        DEC = 0   
        psi = 0
        phi = 0
        m1 *= LAL_MSUN_SI
        m2 *= LAL_MSUN_SI
        hp = self.wvf.generate_a_waveform(m1, m2,
                                          [s1x, s1y, s1z],
                                          [s2x, s2y, s2z],
                                          ecc,
                                          lambda1, lambda2,
                                          iota, phiRef)
        residual = hp
        for k in np.arange(0,len(known_bases)):
            residual -= self.proj(known_bases[k],hp)
        modulus = np.sqrt(np.vdot(residual, residual))
        return modulus

    def compute_modulus_quad(self,paramspoint, known_quad_bases):
        m1, m2 = self.get_m1m2_from_mcq(paramspoint[0],paramspoint[1])
        s1x, s1y, s1z = self.spherical_to_cartesian(paramspoint[2:5]) 
        s2x, s2y, s2z = self.spherical_to_cartesian(paramspoint[5:8]) 
        iota = paramspoint[8]  
        phiRef = paramspoint[9]
        ecc = 0
        if len(paramspoint)==11:
            ecc = paramspoint[10]
        if len(paramspoint)==12:
            lambda1 = paramspoints[arg_newbasis][10]
            lambda2 = paramspoints[arg_newbasis][11]
        f_ref = 0 
        RA = 0    
        DEC = 0  
        psi = 0
        phi = 0
        m1 *= LAL_MSUN_SI
        m2 *= LAL_MSUN_SI
        hp  = self.wvf.generate_a_waveform(m1, m2,
                                           [s1x, s1y, s1z],
                                           [s2x, s2y, s2z],
                                           ecc,
                                           lambda1, lambda2,
                                           iota, phiRef)
        hp_quad_tmp = (np.absolute(hp))**2
        residual = hp_quad_tmp
        for k in np.arange(0,len(known_quad_bases)):
            residual -= self.proj(known_quad_bases[k],hp_quad_tmp)
        modulus = np.sqrt(np.vdot(residual, residual))
        return modulus

    def least_match_waveform_unnormalized(self, paramspoints, known_bases):
        """
        Now generating N=npts waveforms at points that are 
        randomly uniformly distributed in parameter space
        and calculate their inner products with the 1st waveform
        so as to find the best waveform as the new basis
        """
        if self.parallel:
            paramspointslist = paramspoints.tolist()
            #pool = mp.Pool(mp.cpu_count())
            pool = mp.Pool(processes=nprocesses)
            modula = [pool.apply(compute_modulus, args=(paramspoint, known_bases)) for paramspoint in paramspointslist]
            pool.close()
        else:
            npts = len(paramspoints)
            modula = np.zeros(npts)
            for i in np.arange(0,npts):
                paramspoint = paramspoints[i]
                modula[i] = self.compute_modulus(paramspoint, known_bases)
        arg_newbasis = np.argmax(modula) 
        paramspoint = paramspoints[arg_newbasis]
        mass1, mass2 = self.get_m1m2_from_mcq(paramspoints[arg_newbasis][0],paramspoints[arg_newbasis][1])
        mass1 *= LAL_MSUN_SI
        mass2 *= LAL_MSUN_SI
        sp1x, sp1y, sp1z = self.spherical_to_cartesian(paramspoints[arg_newbasis,2:5]) 
        sp2x, sp2y, sp2z = self.spherical_to_cartesian(paramspoints[arg_newbasis,5:8]) 
        inclination = paramspoints[arg_newbasis][8]
        phi_ref = paramspoints[arg_newbasis][9]
        ecc = 0
        if len(paramspoint)==11:
            ecc = paramspoints[arg_newbasis][10]
        if len(paramspoint)==12:
            lambda1 = paramspoints[arg_newbasis][10]
            lambda2 = paramspoints[arg_newbasis][11]
        hp = self.wvf.generate_a_waveform(mass1, mass2,
                                          [sp1x, sp1y, sp1z],
                                          [sp2x, sp2y, sp2z],
                                          ecc,
                                          lambda1, lambda2,
                                          inclination, phi_ref)
        basis_new = self.gram_schmidt(known_bases, hp)
        return np.array([basis_new, paramspoints[arg_newbasis], modula[arg_newbasis]]) # elements, masses&spins, residual mod

    def least_match_quadratic_waveform_unnormalized(self, paramspoints, known_quad_bases):
        if self.parallel:
            paramspointslist = paramspoints.tolist()
            pool = mp.Pool(processes=nprocesses)
            modula = [pool.apply(compute_modulus_quad, args=(paramspoint, known_quad_bases)) for paramspoint in paramspointslist]
            pool.close()
        else:
            npts = len(paramspoints)
            modula = np.zeros(npts)
            for i in np.arange(0,npts):
                paramspoint = paramspoints[i]
                modula[i] = compute_modulus_quad(paramspoint, known_quad_bases)
        arg_newbasis = np.argmax(modula)    
        paramspoint = paramspoints[arg_newbasis]
        mass1, mass2 = self.get_m1m2_from_mcq(paramspoints[arg_newbasis][0],paramspoints[arg_newbasis][1])
        mass1 *= LAL_MSUN_SI
        mass2 *= LAL_MSUN_SI
        sp1x, sp1y, sp1z = self.spherical_to_cartesian(paramspoints[arg_newbasis,2:5]) 
        sp2x, sp2y, sp2z = self.spherical_to_cartesian(paramspoints[arg_newbasis,5:8]) 
        inclination = paramspoints[arg_newbasis][8]
        phi_ref = paramspoints[arg_newbasis][9]
        ecc = 0
        if len(paramspoint)==11:
            ecc = paramspoints[arg_newbasis][10]
        if len(paramspoint)==12:
            lambda1 = paramspoints[arg_newbasis][10]
            lambda2 = paramspoints[arg_newbasis][11]
        hp = self.wvf.generate_a_waveform(mass1, mass2,
                                          [sp1x, sp1y, sp1z],
                                          [sp2x, sp2y, sp2z],
                                          ecc,
                                          lambda1, lambda2,
                                          inclination,
                                          phi_ref)
        hp_quad_new = (np.absolute(hp))**2
        basis_quad_new = self.gram_schmidt(known_quad_bases, hp_quad_new)    
        return np.array([basis_quad_new, paramspoints[arg_newbasis], modula[arg_newbasis]]) # elements, masses&spins, residual mod
    
    def bases_searching_results_unnormalized(self, known_bases, basis_waveforms, params, residual_modula):
        if self.verbose:
            if self.nparams == 10: print("The parameters are Mc, q, s1(mag, theta, phi), s2(mag, theta, phi), iota, and phiRef\n")
            if self.nparams == 11: print("The parameters are Mc, q, s1(mag, theta, phi), s2(mag, theta, phi), iota, phiRef, and eccentricity\n")
            if self.nparams == 12: print("The parameters are Mc, q, s1(mag, theta, phi), s2(mag, theta, phi), iota, phiRef, lambda1, and lambda2\n") 
        for k in np.arange(0,self.nbases-1):
            paramspoints = self.generate_params_points()
            basis_new, params_new, rm_new = self.least_match_waveform_unnormalized(paramspoints, known_bases)
            if self.verbose:
                print("Linear Iter: ", k+1, "and new basis waveform", params_new)
            known_bases= np.append(known_bases, np.array([basis_new]), axis=0)
            params = np.append(params, np.array([params_new]), axis = 0)
            residual_modula = np.append(residual_modula, rm_new)
        np.save(self.outputdir+'/linearbases.npy',known_bases)
        np.save(self.outputdir+'/linearbasiswaveformparams.npy',params)
        return known_bases, params, residual_modula

    def bases_searching_quadratic_results_unnormalized(self,known_quad_bases, basis_waveforms, params_quad, residual_modula):
        for k in np.arange(0,self.nbases_quad-1):
            if self.verbose:
                print("Quadratic Iter: ", k+1)
            paramspoints = self.generate_params_points()
            basis_new, params_new, rm_new = self.least_match_quadratic_waveform_unnormalized(paramspoints, known_quad_bases)
            known_quad_bases= np.append(known_quad_bases, np.array([basis_new]), axis=0)
            params_quad = np.append(params_quad, np.array([params_new]), axis = 0)
            residual_modula = np.append(residual_modula, rm_new)
        np.save(self.outputdir+'/quadraticbases.npy',known_quad_bases)
        np.save(self.outputdir+'/quadraticbasiswaveformparams.npy',params_quad)
        return known_quad_bases, params_quad, residual_modula

    def massrange(self,mc_low, mc_high, q_low, q_high):
        mmin = self.get_m1m2_from_mcq(mc_low,q_high)[1]
        mmax = self.get_m1m2_from_mcq(mc_high,q_high)[0]
        return [mmin, mmax]
    
    def initial_basis(self):
        mc_low = self.intrinsic_params['mc'][0]
        mc_high = self.intrinsic_params['mc'][1], 
        q_low = self.intrinsic_params['q'][0]
        q_high = self.intrinsic_params['q'][1]
        s1sphere_low = self.intrinsic_params['s1sphere'][0]
        s1sphere_high = self.intrinsic_params['s1sphere'][1]
        s2sphere_low = self.intrinsic_params['s2sphere'][0]
        s2sphere_high = self.intrinsic_params['s2sphere'][1]
        ecc_low = self.intrinsic_params['ecc'][0]
        ecc_high = self.intrinsic_params['ecc'][1]
        lambda1_low = self.intrinsic_params['lambda1'][0]
        lambda1_high = self.intrinsic_params['lambda1'][1]
        lambda2_low = self.intrinsic_params['lambda2'][0]
        lambda2_high = self.intrinsic_params['lambda2'][1]
        iota_low = self.intrinsic_params['iota'][0]
        iota_high = self.intrinsic_params['iota'][1]
        phiref_low = self.intrinsic_params['phiref'][0]
        phiref_high = self.intrinsic_params['iota'][1]
        distance = self.distance
        deltaF = self.deltaF
        f_min = self.f_min
        f_max = self.f_max

        self.nparams = NParams[self.approximant]
        
        if self.nparams == 10:
            self.params_low = [mc_low, q_low, s1sphere_low[0], s1sphere_low[1], s1sphere_low[2], s2sphere_low[0], s2sphere_low[1], s2sphere_low[2], iota_low, phiref_low] 
            self.params_high = [mc_high, q_high, s1sphere_high[0], s1sphere_high[1], s1sphere_high[2], s2sphere_high[0], s2sphere_high[1], s2sphere_high[2], iota_high, phiref_high]
            self.params_start = np.array([[mc_low, q_low, s1sphere_low[0], s1sphere_low[1], s1sphere_low[2], s2sphere_low[0], s2sphere_low[1], s2sphere_low[2], 0.33333*np.pi, 1.5*np.pi]])
            self.hp1 = self.generate_a_waveform_from_mcq(mc_low, q_low, spherical_to_cartesian(s1sphere_low), spherical_to_cartesian(s2sphere_low), 0, 0, 0, iota_low, phiref_low)
        elif self.nparams == 11:
            self.params_low = [mc_low, q_low, s1sphere_low[0], s1sphere_low[1], s1sphere_low[2], s2sphere_low[0], s2sphere_low[1], s2sphere_low[2], iota_low, phiref_low, ecc_low] 
            self.params_high = [mc_high, q_high, s1sphere_high[0], s1sphere_high[1], s1sphere_high[2], s2sphere_high[0], s2sphere_high[1], s2sphere_high[2], iota_high, phiref_high, ecc_high]
            self.params_start = np.array([[mc_low, q_low, s1sphere_low[0], s1sphere_low[1], s1sphere_low[2], s2sphere_low[0], s2sphere_low[1], s2sphere_low[2], 0.33333*np.pi, 1.5*np.pi, ecc_low]])
            self.hp1 = self.generate_a_waveform_from_mcq(mc_low, q_low, spherical_to_cartesian(s1sphere_low), spherical_to_cartesian(s2sphere_low), ecc_low, 0, 0, iota_low, phiref_low)
        elif self.nparams == 12:
            self.params_low = [mc_low, q_low, s1sphere_low[0], s1sphere_low[1], s1sphere_low[2], s2sphere_low[0], s2sphere_low[1], s2sphere_low[2], iota_low, phiref_low, lambda1_low, lambda2_low]
            self.params_high = [mc_high, q_high, s1sphere_high[0], s1sphere_high[1], s1sphere_high[2], s2sphere_high[0], s2sphere_high[1], s2sphere_high[2], iota_high, phiref_high, lambda1_high, lambda2_high]
            self.params_start = np.array([[mc_low, q_low, s1sphere_low[0], s1sphere_low[1], s1sphere_low[2], s2sphere_low[0], s2sphere_low[1], s2sphere_low[2], 0.33333*np.pi, 1.5*np.pi, lambda1_low, lambda2_low]])
            
            self.hp1 = self.generate_a_waveform_from_mcq(mc_low, q_low, spherical_to_cartesian(s1sphere_low), spherical_to_cartesian(s2sphere_low), 0, lambda1_low, lambda2_low, iota_low, phiref_low) 
        else:
            raise ValueError 
        
        return 

    def empnodes(self, ndim, known_bases):
        """
        Here known_bases is the full copy known_bases_copy. Its length is equal to or longer than ndim.
        """
        emp_nodes = np.arange(0,ndim)*100000000
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
        #print(len(emp_nodes), "\nDuplicates indices:", dup)
        emp_nodes = np.unique(emp_nodes)
        ndim = len(emp_nodes)
        #print(len(emp_nodes), "\n", emp_nodes)
        V = np.transpose(known_bases[0:ndim, emp_nodes])
        inverse_V = np.linalg.pinv(V)
        return np.array([ndim, inverse_V, emp_nodes])

    def surroerror(self, ndim, inverse_V, emp_nodes, known_bases, test_mc, test_q, test_s1, test_s2, test_ecc, test_lambda1, test_lambda2, test_iota, test_phiref):
        hp_test = self.generate_a_waveform_from_mcq(test_mc, test_q, test_s1, test_s2, test_ecc, test_lambda1, test_lambda2, test_iota, test_phiref)
        Ci = np.dot(inverse_V, hp_test[emp_nodes])
        interpolantA = np.zeros(len(hp_test))+np.zeros(len(hp_test))*1j
        #ndim = len(known_bases)
        for j in np.arange(0, ndim):
            tmp = np.multiply(Ci[j], known_bases[j])
            interpolantA += tmp
        surro = (1-overlap_of_two_waveforms(hp_test, interpolantA))*deltaF
        return surro

    def surros(ndim, inverse_V, emp_nodes, known_bases): 
        """
        Here known_bases is known_bases_copy
        """
        test_points = self.generate_params_points()
        surros = np.zeros(nts)
        count = 0
        for i in np.arange(0,nts):
            test_mc =  test_points[i,0]
            test_q = test_points[i,1]
            test_s1 = self.spherical_to_cartesian(test_points[i,2:5])
            test_s2 = self.spherical_to_cartesian(test_points[i,5:8])
            test_iota = test_points[i,8]
            test_phiref = test_points[i,9]
            test_ecc = 0
            test_lambda1 = 0
            test_lambda2 = 0
            if nparams == 11:
                test_ecc = test_points[i,10]
            if nparams == 12: 
                test_lambda1 = test_points[i,10]
                test_lambda2 = test_points[i,11]
            surros[i] = self.surroerror(ndim, inverse_V, emp_nodes, known_bases[0:ndim], test_mc, test_q, test_s1, test_s2, test_ecc, test_lambda1, test_lambda2, test_iota, test_phiref)
            if (surros[i] > self.tolerance):
                count = count+1
        if self.verbose:
            print(ndim, "basis elements gave", count, "bad points of surrogate error > ", self.tolerance)
        if count == 0:
            val = 0
        else:
            val = 1
        return val

    def roqs(self, known_bases_copy): 
        for num in np.arange(self.ndimlow, self.ndimhigh, self.ndimstepsize):
            ndim, inverse_V, emp_nodes = self.empnodes(num, known_bases_copy)
            if self.surros(ndim, inverse_V, emp_nodes, known_bases_copy) == 0:
                b_linear = np.dot(np.transpose(known_bases_copy[0:ndim]),inverse_V)
                f_linear = self.freq[emp_nodes]
                np.save(self.outputdir+'/B_linear.npy',np.transpose(b_linear))
                np.save(self.outputdir+'/fnodes_linear.npy',f_linear)
                if self.verbose:
                    print("Number of linear basis elements is ", ndim, "and the linear ROQ data are saved in B_linear.npy")
                break
        return

    def testrep(self, b_linear, emp_nodes, test_mc, test_q, test_s1, test_s2, test_ecc, test_lambda1, test_lambda2, test_iota, test_phiref):
        hp_test = self.generate_a_waveform_from_mcq(test_mc, test_q, test_s1, test_s2, test_ecc, test_lambda1, test_lambda2, test_iota, test_phiref)
        hp_test_emp = hp_test[emp_nodes]
        hp_rep = np.dot(b_linear,hp_test_emp)
        diff = hp_rep - hp_test
        rep_error = diff/np.sqrt(np.vdot(hp_test,hp_test))
        freq = np.arange(f_min,f_max,deltaF)
        #TODO return figure handle
        plt.figure(figsize=(15,9))
        plt.plot(freq, np.real(rep_error), label='Real part of h+') 
        plt.plot(freq, np.imag(rep_error), label='Imaginary part of h+')
        plt.xlabel('Frequency')
        plt.ylabel('Fractional Representation Error')
        plt.title('Rep Error with np.linalg.pinv()')
        plt.legend(loc=0)
        plt.show()
        return

    def empnodes_quad(self, ndim_quad, known_quad_bases):
        emp_nodes_quad = np.arange(0,ndim_quad)*100000000
        emp_nodes_quad[0] = np.argmax(np.absolute(known_quad_bases[0]))
        c1_quad = known_quad_bases[1,emp_nodes_quad[0]]/known_quad_bases[0,1]
        interp1_quad = np.multiply(c1_quad,known_quad_bases[0])
        diff1_quad = interp1_quad - known_quad_bases[1]
        r1_quad = np.absolute(diff1_quad)
        emp_nodes_quad[1] = np.argmax(r1_quad)
        for k in np.arange(2,ndim_quad):
            emp_tmp_quad = emp_nodes_quad[0:k]
            Vtmp_quad = np.transpose(known_quad_bases[0:k,emp_tmp_quad])
            inverse_Vtmp_quad = np.linalg.pinv(Vtmp_quad)
            e_to_interp_quad = known_quad_bases[k]
            Ci_quad = np.dot(inverse_Vtmp_quad, e_to_interp_quad[emp_tmp_quad])
            interpolantA_quad = np.zeros(len(known_quad_bases[k]))+np.zeros(len(known_quad_bases[k]))*1j
            for j in np.arange(0, k):
                tmp_quad = np.multiply(Ci_quad[j], known_quad_bases[j])
                interpolantA_quad += tmp_quad
            diff_quad = interpolantA_quad - known_quad_bases[k]
            r_quad = np.absolute(diff_quad)
            emp_nodes_quad[k] = np.argmax(r_quad)
            emp_nodes_quad = sorted(emp_nodes_quad)
        u_quad, c_quad = np.unique(emp_nodes_quad, return_counts=True)
        dup_quad = u_quad[c_quad > 1]
        #print(len(emp_nodes_quad), "\nduplicates quad indices:", dup_quad)
        emp_nodes_quad = np.unique(emp_nodes_quad)
        ndim_quad = len(emp_nodes_quad)
        #print(len(emp_nodes_quad), "\n", emp_nodes_quad)
        V_quad = np.transpose(known_quad_bases[0:ndim_quad,emp_nodes_quad])
        inverse_V_quad = np.linalg.pinv(V_quad)
        return np.array([ndim_quad, inverse_V_quad, emp_nodes_quad])

    def surroerror_quad(self, ndim_quad, inverse_V_quad, emp_nodes_quad, known_quad_bases, test_mc_quad, test_q_quad, test_s1_quad, test_s2_quad, test_ecc_quad, test_lambda1_quad, test_lambda2_quad, test_iota_quad, test_phiref_quad):
        hp = self.generate_a_waveform_from_mcq(test_mc_quad, test_q_quad, test_s1_quad, test_s2_quad, test_ecc_quad, test_lambda1_quad, test_lambda2_quad, test_iota_quad, test_phiref_quad)
        hp_test_quad = (np.absolute(hp))**2
        Ci_quad = np.dot(inverse_V_quad, hp_test_quad[emp_nodes_quad])
        interpolantA_quad = np.zeros(len(hp_test_quad))+np.zeros(len(hp_test_quad))*1j    
        #ndim_quad = len(known_quad_bases)
        for j in np.arange(0, ndim_quad):
            tmp_quad = np.multiply(Ci_quad[j], known_quad_bases[j])
            interpolantA_quad += tmp_quad
        surro_quad = (1-overlap_of_two_waveforms(hp_test_quad, interpolantA_quad))*deltaF
        return surro_quad

    def surros_quad(self, tolerance_quad, ndim_quad, inverse_V_quad, emp_nodes_quad, known_quad_bases):
        test_points = self.generate_params_points()
        surros = np.zeros(nts)
        count = 0
        for i in np.arange(0,nts):
            test_mc_quad =  test_points[i,0]
            test_q_quad = test_points[i,1]
            test_s1_quad = spherical_to_cartesian(test_points[i,2:5])
            test_s2_quad = spherical_to_cartesian(test_points[i,5:8])
            test_iota_quad = test_points[i,8]
            test_phiref_quad = test_points[i,9]
            test_ecc_quad = 0
            test_lambda1_quad = 0
            test_lambda2_quad = 0
            if self.nparams == 11:
                test_ecc_quad = test_points[i,10]
            if self.nparams == 12: 
                test_lambda1_quad = test_points[i,10]
                test_lambda2_quad = test_points[i,11]
            surros[i] = self.surroerror_quad(ndim_quad, inverse_V_quad, emp_nodes_quad, known_quad_bases[0:ndim_quad], test_mc_quad, test_q_quad, test_s1_quad, test_s2_quad, test_ecc_quad, test_lambda1_quad, test_lambda2_quad, test_iota_quad, test_phiref_quad)
            if (surros[i] > tolerance_quad):
                count = count+1
        if self.verbose:
            print(ndim_quad, "basis elements gave", count, "bad points of surrogate error > ", tolerance_quad)
        if count == 0:
            val =0
        else:
            val = 1
        return val

    def roqs_quad(self, tolerance_quad, freq,  ndimlow_quad, ndimhigh_quad, ndimstepsize_quad, known_quad_bases_copy):
        for num in np.arange(ndimlow_quad, ndimhigh_quad, ndimstepsize_quad):
            ndim_quad, inverse_V_quad, emp_nodes_quad = empnodes_quad(num, known_quad_bases_copy)
            if self.surros_quad(tolerance_quad, ndim_quad, inverse_V_quad, emp_nodes_quad, known_quad_bases_copy)==0:
                b_quad = np.dot(np.transpose(known_quad_bases_copy[0:ndim_quad]), inverse_V_quad)
                f_quad = freq[emp_nodes_quad]
                np.save(self.outputdir+'/B_quadratic.npy', np.transpose(b_quad))
                np.save(self.outputdir+'/fnodes_quadratic.npy', f_quad)
                if self.verbose:
                    print("Number of quadratic basis elements is ", ndim_quad, "and the linear ROQ data save in B_quadratic.npy")
                break
        return

    def testrep_quad(self, b_quad, emp_nodes_quad, test_mc_quad, test_q_quad, test_s1_quad, test_s2_quad, test_ecc_quad, test_lambda1_quad, test_lambda2_quad, test_iota_quad, test_phiref_quad):
        hp = self.generate_a_waveform_from_mcq(test_mc_quad, test_q_quad, test_s1_quad, test_s2_quad, test_ecc_quad, test_lambda1_quad, test_lambda2_quad, test_iota_quad, test_phiref_quad)
        hp_test_quad = (np.absolute(hp))**2
        hp_test_quad_emp = hp_test_quad[emp_nodes_quad]
        hp_rep_quad = np.dot(b_quad,hp_test_quad_emp)
        diff_quad = hp_rep_quad - hp_test_quad
        rep_error_quad = diff_quad/np.vdot(hp_test_quad,hp_test_quad)**0.5
        freq = np.arange(f_min,f_max,deltaF)
        #TODO return figure handle
        plt.figure(figsize=(15,9))
        plt.plot(freq, np.real(rep_error_quad))
        plt.xlabel('Frequency')
        plt.ylabel('Fractional Representation Error for Quadratic')
        plt.title('Rep Error with np.linalg.pinv()')
        plt.show()
        return

    def surros_of_test_samples(self, nsamples, b_linear, emp_nodes):
        nts = nsamples
        ndim = len(emp_nodes)
        test_points = self.generate_params_points()
        surros = np.zeros(nts)
        for i in np.arange(0,nts):
            test_mc = test_points[i,0]
            test_q = test_points[i,1]
            test_s1 = spherical_to_cartesian(test_points[i,2:5])
            test_s2 = spherical_to_cartesian(test_points[i,5:8])
            test_iota = test_points[i,8]
            test_phiref = test_points[i,9]
            test_ecc = 0
            test_lambda1 = 0
            test_lambda2 = 0
            if self.nparams == 11:
                test_ecc = test_points[i,10]
            if self.nparams == 12: 
                test_lambda1 = test_points[i,10]
                test_lambda2 = test_points[i,11]
            hp_test = self.generate_a_waveform_from_mcq(test_mc, test_q, test_s1, test_s2, test_ecc, test_lambda1, test_lambda2, test_iota, test_phiref)
            hp_test_emp = hp_test[emp_nodes]
            hp_rep = np.dot(b_linear,hp_test_emp) 
            surros[i] = (1-overlap_of_two_waveforms(hp_test, hp_rep))*deltaF
            if self.verbose:
                if (surros[i] > tolerance):
                    print("iter", i, surros[i], test_points[i])
                if i%100==0:
                    print("iter", i, surros[i])
        return surros


if __name__ == '__main__':

    # example
    pyroq = PyROQ(outpudir='./test')

    # Search for linear basis elements to build and save linear ROQ data in the local directory.
    hp1 = pyroq.hp1
    known_bases_start = np.array([hp1/np.sqrt(np.vdot(hp1,hp1))])
    basis_waveforms_start = np.array([hp1])
    residual_modula_start = np.array([0.0])
    known_bases, params, residual_modula = pyroq.bases_searching_results_unnormalized(known_bases_start, basis_waveforms_start, residual_modula_start)
    
    print(known_bases.shape, residual_modula)

    # Create ROQ (save to file).
    known_bases = np.load(pyroq.outpudir+'/linearbases.npy')
    pyroq.roqs(known_bases)

    # Check
    fnodes_linear = np.load(pyroq.outpudir+'/fnodes_linear.npy')
    b_linear = np.transpose(np.load(pyroq.outpudir+'/B_linear.npy'))
    ndim = b_linear.shape[1]
    freq = pyroq.freq # np.arange(f_min, f_max, deltaF)
    emp_nodes = np.searchsorted(freq, fnodes_linear)
    print(b_linear)
    print("emp_nodes", emp_nodes)

    test_mc = 25
    test_q = 2
    test_s1 = [0.,0.2,-0.]
    test_s2 = [0.,0.15,-0.1]
    test_ecc = 0
    test_lambda1 = 0
    test_lambda2 = 0
    test_iota = 1.9
    test_phiref = 0.6
    pyroq.testrep(b_linear, emp_nodes, test_mc, test_q, test_s1, test_s2, test_ecc, test_lambda1, test_lambda2, test_iota, test_phiref)

    # Test nsamples random samples in parameter space to see their representation surrogate errors
    nsamples = 1000 
    surros = pyroq.surros_of_test_samples(nsamples, b_linear, emp_nodes)

    plt.figure(figsize=(15,9))
    plt.semilogy(surros,'o',color='black')
    plt.xlabel("Number of Random Test Points")
    plt.ylabel("Surrogate Error")
    plt.title("IMRPhenomPv2")
    plt.savefig("SurrogateErrorsRandomTestPoints.png")
    plt.show()

    # Search for quadratic basis elements to build & save quadratic ROQ data.
    hp1_quad = (np.absolute(hp1))**2
    known_quad_bases_start = np.array([hp1_quad/np.sqrt(np.vdot(hp1_quad,hp1_quad))])
    basis_waveforms_quad_start = np.array([hp1_quad])
    residual_modula_start = np.array([0.0])
    known_quad_bases,params_quad,residual_modula_quad = pyroq.bases_searching_quadratic_results_unnormalized(known_quad_bases_start, basis_waveforms_quad_start, residual_modula_start)
    known_quad_bases_copy = known_quad_bases

    known_quad_bases = np.load(pyroq.outpurdir+'/quadraticbases.npy')
    pyroq.roqs_quad(known_quad_bases)

    fnodes_quad = np.load(pyroq.outpurdir+'/fnodes_quadratic.npy')
    b_quad = np.transpose(np.load(pyroq.outpurdir+'/B_quadratic.npy'))
    ndim_quad = b_quad.shape[1]
    freq = pyroq.freq # np.arange(f_min, f_max, deltaF)
    emp_nodes_quad = np.searchsorted(freq, fnodes_quad)

    # Check
    test_mc_quad = 22
    test_q_quad = 1.2
    test_s1_quad = [0.0, 0.1, 0.0]
    test_s2_quad = [0.0, 0.0, 0.0]
    test_ecc_quad = 0
    test_lambda1_quad = 0
    test_lambda2_quad = 0
    test_iota_quad = 1.9
    test_phiref_quad = 0.6

    pyroq.testrep_quad(b_quad, emp_nodes_quad, test_mc_quad, test_q_quad, test_s1_quad, test_s2_quad, test_ecc_quad, test_lambda1_quad, test_lambda2_quad, test_iota_quad, test_phiref_quad)


