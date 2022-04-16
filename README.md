# PyROQ
This is the repository for the PyROQ code that builds ROQ data for the long duration waveforms in LIGO and can be generalized to LISA as well. The code is written in Python. If you have any comments, please email hong.qi@ligo.org.

Please cite [our paper](https://arxiv.org/abs/2009.13812) if you use our code in your research. 

SB (sebastiano.bernuzzi@unin-jena.de) 03/2022:
   * Forked PyRQO version 0.1.26 repo
   * Added support for [TEOBResumS GIOTTO](https://bitbucket.org/eob_ihes/teobresums/src/master/) and MLW-BNS
   * Refactored code
     - Introduced PyROQ class
     - Simplified code/reduced duplication
     - Added waveform wrapper classes
     - Changed parameter management