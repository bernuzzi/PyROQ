# PyROQ
This repository implements a streamlined version of the PyROQ code, branched in March 2022 from the [main repository](https://github.com/qihongcat/PyROQ).
Please cite the [PyROQ paper](https://arxiv.org/abs/2009.13812) if you use this code in your research. 

# Usage

Starting from the `PyROQ_repo_directory/PyROQ` location, 
the package can be installed using the command:

    python setup.py install

Once  `PyROQ` is  installed, a simple example can be run by:

    python -m PyROQ --config-file config_files/test_config_IMRPv2_GW150914_LVK.ini
    
The full list  of options can be printed using:

    python -m PyROQ --help

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
