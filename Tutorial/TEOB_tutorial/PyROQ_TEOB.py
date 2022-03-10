# Standard python imports
import h5py, numpy, random, scipy, warnings
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
    'mc'      : [20, 30]                                  ,
    'q'       : [1, 2]                                    ,
    's1sphere': [[0, 0, 0], [0.2, numpy.pi, 2.0*numpy.pi]],
    's2sphere': [[0, 0, 0], [0.2, numpy.pi, 2.0*numpy.pi]],
    'ecc'     : [0.0, 0.2]                                ,
    'lambda1' : [0, 1000]                                 ,
    'lambda2' : [0, 1000]                                 ,
    'iota'    : [0, numpy.pi]                             ,
    'phiref' : [0, 2*numpy.pi]                           ,
}

# Frequency axis on which the interpolant will be constructed
f_min, f_max, deltaF = 20, 1024, 1/4.
approximant = 'teobresums-giotto-FD'

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
          # For diagnostic testing, 30 -100 is fine. 
          # For real ROQs computation, this can be 300 to 2000, roughly comparable to the number of basis elments.
          # What value to choose depends on the nature of the waveform, such as how many features it has. 
          # It also depends on the parameter space and the signal length. 
        
nbases = 80 # Specify the number of linear basis elements. Put your estimation here for the chunk of parameter space.
ndimlow = 40 # Your estimation of fewest basis elements needed for this chunk of parameter space.
ndimhigh = nbases+1 
ndimstepsize = 10 # Number of linear basis elements increament to check if the basis satisfies the tolerance.
tolerance = 1e-8 # Surrogage error threshold for linear basis elements

nbases_quad = 80 # Specify the number of quadratic basis elements, depending on the tolerance_quad, usually two thirds of that for linear basis
ndimlow_quad = 20
ndimhigh_quad = nbases_quad+1
ndimstepsize_quad = 10
tolerance_quad = 1e-10 # Surrogage error threshold for quadratic basis elements

#############################################################
# Below this point, ideally no parameter should be changed. #
#############################################################

# Dummy value, distance does not enter the interolants construction
distance = 10 * LAL_PC_SI * 1.0e6  # 10 Mpc is default 

waveFlags = pyroq.eob_parameters()
print("mass-min, mass-max: ", pyroq.massrange(intrinsic_params['mc'][0], intrinsic_params['mc'][1], intrinsic_params['q'][0], intrinsic_params['q'][1]))

# Create the ROQ initial basis
freq = numpy.arange(f_min,f_max,deltaF)
nparams, params_low, params_high, params_start, hp1 = pyroq.initial_basis(intrinsic_params['mc'][0], intrinsic_params['mc'][1], 
                                                                          intrinsic_params['q'][0], intrinsic_params['q'][1], 
                                                                          intrinsic_params['s1sphere'][0], intrinsic_params['s1sphere'][1],
                                                                          intrinsic_params['s2sphere'][0], intrinsic_params['s2sphere'][1],
                                                                          intrinsic_params['ecc'][0], intrinsic_params['ecc'][1],
                                                                          intrinsic_params['lambda1'][0], intrinsic_params['lambda1'][1],
                                                                          intrinsic_params['lambda2'][0], intrinsic_params['lambda2'][1],
                                                                          intrinsic_params['iota'][0], intrinsic_params['iota'][1],
                                                                          intrinsic_params['phiref'][0], intrinsic_params['iota'][1],
                                                                          distance, 
                                                                          deltaF, f_min, f_max, 
                                                                          waveFlags, approximant)

print('WHAT AM I UNPACKING????????')

known_bases_start = numpy.array([hp1/numpy.sqrt(numpy.vdot(hp1,hp1))])
basis_waveforms_start = numpy.array([hp1])
residual_modula_start = numpy.array([0.0])
known_bases, params, residual_modula = pyroq.bases_searching_results_unnormalized(parallel, nprocesses, npts, nparams, nbases, known_bases_start, basis_waveforms_start, params_start, residual_modula_start, params_low, params_high, distance, deltaF, f_min, f_max, waveFlags, approximant)
print(known_bases.shape, residual_modula)
