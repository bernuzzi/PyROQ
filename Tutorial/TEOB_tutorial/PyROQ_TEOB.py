# Standard python imports
import h5py, numpy, os, random, scipy, warnings
import matplotlib, matplotlib.pylab as pylab, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# LAL imports
import lal
import lalsimulation
from lal.lal import PC_SI as LAL_PC_SI

# Package imports
import PyROQ.pyroq as pyroq
#import pyroq as pyroq

warnings.filterwarnings('ignore')
plot_params = {'legend.fontsize': 'x-large',
               'figure.figsize' : (15, 9)  ,
               'axes.labelsize' : 'x-large',
               'axes.titlesize' :'x-large' ,
               'xtick.labelsize':'x-large' ,
               'ytick.labelsize':'x-large' }
pylab.rcParams.update(plot_params)

# Intrinsic parameter space on which the interpolants will be constructed
intrinsic_params = {
    'mc'      : [45, 46]                                  ,
    'q'       : [1, 1.2]                                  ,
    's1sphere': [[0, 0, 0], [0.2, numpy.pi, 2.0*numpy.pi]],
    's2sphere': [[0, 0, 0], [0.2, numpy.pi, 2.0*numpy.pi]],
    'ecc'     : [0.0, 0.0]                                ,
    'lambda1' : [0, 1000]                                 ,
    'lambda2' : [0, 1000]                                 ,
    'iota'    : [0, numpy.pi]                             ,
    'phiref'  : [0, 2*numpy.pi]                           ,
}

# Frequency axis on which the interpolant will be constructed
f_min, f_max, deltaF = 20, 512, 1/4.
approximant = 'teobresums-giotto-FD'

run_tag = 'test_freqs'
if not os.path.exists(run_tag): os.makedirs(run_tag)
# Computing parameters
parallel = 0 # The parallel=1 will turn on multiprocesses to search for a new basis. To turn it off, set it to be 0.
             # Do not turn it on if the waveform generation is not slow compared to data reading and writing to files.
             # This is more useful when each waveform takes larger than 0.01 sec to generate.
nprocesses = 4 # Set the number of parallel processes when searching for a new basis.  nprocesses=mp.cpu_count()

# Interpolants construction parameters
nts = 123 # Number of random test waveforms
          # For diagnostics, 1000 is fine.
          # For real ROQs calculation, set it to be 1000000.

npts = 80 # Specify the number of points for each search for a new basis element
          # For diagnostic testing, 30-100 is fine. 
          # For real ROQs computation, this can be 300 to 2000, roughly comparable to the number of basis elments.
          # What value to choose depends on the nature of the waveform, such as how many features it has. 
          # It also depends on the parameter space and the signal length. 
        
nbases = 30 # Specify the number of linear basis elements. Put your estimation here for the chunk of parameter space.
ndimlow = 20 # Your estimation of fewest basis elements needed for this chunk of parameter space.
ndimhigh = nbases+1 
ndimstepsize = 1 # Number of linear basis elements increment to check if the basis satisfies the tolerance.
tolerance = 1e-4 # Surrogage error threshold for linear basis elements

nbases_quad = 30 # Specify the number of quadratic basis elements, depending on the tolerance_quad, usually two thirds of that for linear basis
ndimlow_quad = 20
ndimhigh_quad = nbases_quad+1
ndimstepsize_quad = 1
tolerance_quad = 1e-5 # Surrogage error threshold for quadratic basis elements

plot_only = 0

#############################################################
# Below this point, ideally no parameter should be changed. #
#############################################################

# Dummy value, distance does not enter the interolants construction
distance = 10 * LAL_PC_SI * 1.0e6  # 10 Mpc is default 

waveFlags = pyroq.eob_parameters()
print("mass-min, mass-max: ", pyroq.massrange(intrinsic_params['mc'][0], intrinsic_params['mc'][1], intrinsic_params['q'][0], intrinsic_params['q'][1]))

# Create the ROQ initial basis
freq = numpy.arange(f_min, f_max, deltaF)
nparams, params_low, params_high, params_start, hp1 = pyroq.initial_basis(intrinsic_params['mc'][0],       intrinsic_params['mc'][1], 
                                                                          intrinsic_params['q'][0],        intrinsic_params['q'][1], 
                                                                          intrinsic_params['s1sphere'][0], intrinsic_params['s1sphere'][1],
                                                                          intrinsic_params['s2sphere'][0], intrinsic_params['s2sphere'][1],
                                                                          intrinsic_params['ecc'][0],      intrinsic_params['ecc'][1],
                                                                          intrinsic_params['lambda1'][0],  intrinsic_params['lambda1'][1],
                                                                          intrinsic_params['lambda2'][0],  intrinsic_params['lambda2'][1],
                                                                          intrinsic_params['iota'][0],     intrinsic_params['iota'][1],
                                                                          intrinsic_params['phiref'][0],   intrinsic_params['iota'][1],
                                                                          distance, 
                                                                          deltaF, f_min, f_max, 
                                                                          waveFlags, approximant)

print('WHAT AM I UNPACKING????????')

if not plot_only:
    known_bases_start = numpy.array([hp1/numpy.sqrt(numpy.vdot(hp1,hp1))])
    basis_waveforms_start = numpy.array([hp1])
    residual_modula_start = numpy.array([0.0])
    known_bases, params, residual_modula = pyroq.bases_searching_results_unnormalized(parallel, nprocesses, npts, nparams, nbases, known_bases_start, basis_waveforms_start, params_start, residual_modula_start, params_low, params_high, distance, deltaF, f_min, f_max, waveFlags, approximant)
    print(known_bases.shape, residual_modula)
    
    known_bases = numpy.load('./linearbases.npy')
    pyroq.roqs(tolerance, freq, ndimlow, ndimhigh, ndimstepsize, known_bases, nts, nparams, params_low, params_high, distance, deltaF, f_min, f_max, waveFlags, approximant)
    fnodes_linear, b_linear = numpy.load('./fnodes_linear.npy'), numpy.transpose(numpy.load('./B_linear.npy'))

    os.system('mv ./linearbases.npy ./linearbasiswaveformparams.npy ./fnodes_linear.npy ./B_linear.npy {}/.'.format(run_tag)) 
else:
    fnodes_linear, b_linear = numpy.load(os.path.join(run_tag,'fnodes_linear.npy')), numpy.transpose(numpy.load(os.path.join(run_tag,'B_linear.npy')))

emp_nodes_linear = numpy.searchsorted(freq, fnodes_linear)
print('Linear interpolant dimensions:', b_linear.shape)
print('Indices of new linear frequency nodes: ', emp_nodes_linear)
print('Linear basis reduction factor: (Original freqs [{}]) / (New freqs [{}]) = {}'.format(len(freq), len(fnodes_linear), len(freq)/len(fnodes_linear)))

test_mc      = 45.5
test_q       = 1.1
test_s1      = [0.1,0.2,-0.]
test_s2      = [0.1,0.15,-0.1]
test_ecc     = 0
test_lambda1 = 200
test_lambda2 = 200
test_iota    = 1.9
test_phiref  = 0.6

pyroq.testrep(b_linear, emp_nodes_linear, test_mc, test_q, test_s1, test_s2, test_ecc, test_lambda1, test_lambda2, test_iota, test_phiref, distance, deltaF, f_min, f_max, waveFlags, approximant)

nsamples = 100 # testing nsamples random samples in parameter space to see their representation surrogate errors
surros = pyroq.surros_of_test_samples(nsamples, nparams, params_low, params_high, tolerance, b_linear, emp_nodes_linear, distance, deltaF, f_min, f_max, waveFlags, approximant)

plt.figure(figsize=(15,9))
plt.semilogy(surros,'o',color='black')
plt.xlabel("Number of Random Test Points")
plt.ylabel("Surrogate Error")
plt.title("TEOB_FD")
plt.savefig(os.path.join(run_tag,"SurrogateErrorsRandomTestPoints.png"))

# Quadratic basis

if not plot_only:

    hp1_quad = (numpy.absolute(hp1))**2
    known_quad_bases_start = numpy.array([hp1_quad/numpy.sqrt(numpy.vdot(hp1_quad,hp1_quad))])
    basis_waveforms_quad_start = numpy.array([hp1_quad])
    residual_modula_start = numpy.array([0.0])
    known_quad_bases, params_quad, residual_modula_quad = pyroq.bases_searching_quadratic_results_unnormalized(parallel, nprocesses, npts, nparams, nbases_quad, known_quad_bases_start, basis_waveforms_quad_start, params_start, residual_modula_start, params_low, params_high, distance, deltaF, f_min, f_max, waveFlags, approximant)

    known_quad_bases = numpy.load('./quadraticbases.npy')
    pyroq.roqs_quad(tolerance_quad, freq, ndimlow_quad, ndimhigh_quad, ndimstepsize_quad, known_quad_bases, nts, nparams, params_low, params_high, distance, deltaF, f_min, f_max, waveFlags, approximant)

    fnodes_quad,b_quad = numpy.load('./fnodes_quadratic.npy'), numpy.transpose(numpy.load('./B_quadratic.npy'))
    os.system('mv ./fnodes_quadratic.npy ./B_quadratic.npy ./quadraticbases.npy ./quadraticbasiswaveformparams.npy {}/.'.format(run_tag)) 
else:
    fnodes_quad,b_quad = numpy.load(os.path.join(run_tag,'fnodes_quadratic.npy')), numpy.transpose(numpy.load(os.path.join(run_tag,'B_quadratic.npy')))

ndim_quad      = b_quad.shape[1]
emp_nodes_quad = numpy.searchsorted(freq, fnodes_quad)
print('Indices of new quadratic frequency nodes: ', emp_nodes_quad)
print('Quadratic basis reduction factor: (Original freqs [{}]) / (New freqs [{}]) = {}'.format(len(freq), len(fnodes_quad), len(freq)/len(fnodes_quad)))

pyroq.testrep_quad(b_quad, emp_nodes_quad, test_mc, test_q, test_s1, test_s2, test_ecc, test_lambda1, test_lambda2, test_iota, test_phiref, distance, deltaF, f_min, f_max, waveFlags, approximant)
os.system('mv ./testrep.png ./testrepquad.png {}/.'.format(run_tag)) 
