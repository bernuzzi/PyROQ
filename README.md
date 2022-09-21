*IMPORTANT* : THE DEVELOPMENT OF THIS CODE CONTINUES AS [JenapyROQ](https://github.com/GCArullo/JenpyROQ).


# PyROQ

This repository implements a streamlined version of the PyROQ code, branched in March 2022 from the [main repository](https://github.com/qihongcat/PyROQ).
Please cite the [PyROQ paper](https://arxiv.org/abs/2009.13812) if you use this code in your research. 

# Installation and usage

Starting from the `PyROQ` location, 
the package can be installed using the command:

    python setup.py install

Once  `PyROQ` is  installed, it is possible to construct an ROQ approximant through a configuration file and running the main routine of the package:

    python -m PyROQ --config-file config.ini

The user can see the full list of options at:

    python -m PyROQ --help

A simple example can be run by:

    python -m PyROQ --config-file config_files/Test_configs/test_config_IMRPv2_GW150914_LVK.ini

Other examples are available in the `config_files` directory, see the relative [README file](https://github.com/bernuzzi/PyROQ/blob/master/config_files/Test_configs/README.md).

For MPI-based parallelisation, the command should be modified as follows:
    
    mpiexec -n NTASKS python -m PyROQ --config-file config_files/Test_configs/test_config_IMRPv2_GW150914_LVK.ini
    
Where `NTASKS` corresponds to the requested number of parallel tasks. 
Moreover, the config file should specify the related flag `parallel=2` for MPI-based parallelisation and `n-processes` should correspond to NTASKS.  


# Output

The run directory will automatically contain a copy of the configuration file, git information and the screen output, stored under `PyROQ.log`.

Preselection basis and related parameters, together with the enriched basis, its related parameters, the basis interpolant and empirical nodes are stored at each step of the enrichment loop under the `ROQ_data` directory.

Several diagnostic plots (basis parameters, frequency nodes, outliers and error evolution, a single test waveform comparison and validation tests) are stored under the `Plots` directory.

# Algorithm description

MISSING
        
# Dependencies

The package depends on standard Python libraries, except for: `numpy` for numeric computation, `h5py` for data storing and `matplotlib` for plotting. Moreover, if MPI-based parallelisation is requested, the package has an additional dependency on `mpi4py`.

# Development history

SB (sebastiano.bernuzzi@uni-jena.de) 03/2022:
   * Forked PyRQO version 0.1.26 repo
   * Added support for [TEOBResumS GIOTTO](https://bitbucket.org/eob_ihes/teobresums/src/master/) and MLW-BNS
   * Refactored code
     - Introduced PyROQ class
     - Simplified code/reduced duplication
     - Added waveform wrapper classes
     - Changed parameter management

GC (gregorio.carullo@uni-jena.de) 05/2022:
  * Debugged and simplified `refactored` branch.
  * Switched to config file usage.

GC (gregorio.carullo@uni-jena.de) 05/2022:

  * Implemented PyROQ algorithm described in PyROQ paper: pre-selection loop and subsequent enrichment cycles.
  * Allow user to determine enrichment cycles properties,
  * (Almost) maximally streamline code and move logically separated functions to specific files.
  * Parallelise linear and quadratic, add more parallelisation steps where possible.
  * Improve post-processing and input storage (git info, config file, stdout/stderr).
  
MB (matteo.breschi@uni-jena.de) 05/2022:
  
  * Introduce logger
  * Implement MPI-based parallelisation and unify pool usage
  * Extend setup.py, improve packaging and include main functionalities
