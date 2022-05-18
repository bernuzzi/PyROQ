import numpy as np
    
## General linear algebra routines

def proj(u, v):
    """
    Calculating the projection of complex vector v on complex vector u.
    Note: this algorithm assumes u isn't zero.
    """
    return u * np.vdot(v,u)

def normalise_vector(vec):

    return vec/np.sqrt(np.vdot(vec,vec))

def gram_schmidt(bases, vec):
    """
    Calculating the normalized residual (= a new basis term) of a vector vec from known bases.
    """
    for i in np.arange(0,len(bases)):
        vec = vec - proj(bases[i], vec)
    return normalise_vector(vec)

def overlap_of_two_waveforms(wf1, wf2, deltaF, error_version):
    """
        Calculating overlap (FIXME: change to a more representative name) of two waveforms.
    """
    
    # From the forked master version of the public PyROQ: https://github.com/qihongcat/PyROQ/blob/cb6350751dcff303957ace5ac83e6ff6e265a9c7/Code/PyROQ/pyroq.py#L40
    if(error_version=='v1'):
        wf1norm = normalise_vector(wf1)
        wf2norm = normalise_vector(wf2)
        measure = (1-np.real(np.vdot(wf1norm, wf2norm)))*deltaF
    # From the PyROQ paper: https://arxiv.org/abs/2009.13812
    elif(error_version=='v2'):
        diff    = wf1 - wf2
        measure = np.real(np.vdot(diff, diff))*deltaF
    # From the forked master version of the public PyROQ (commented): https://github.com/qihongcat/PyROQ/blob/cb6350751dcff303957ace5ac83e6ff6e265a9c7/Code/PyROQ/pyroq.py#L39
    elif(error_version=='v3'):
        diff    = wf1 - wf2
        measure = 1 - 0.5*np.real(np.vdot(diff, diff))
    # Same as 'v3', but without the (1-0.5*) factor
    elif(error_version=='v4'):
        diff    = wf1 - wf2
        measure = np.real(np.vdot(diff, diff))
    else:
        raise VersionError
    
    return measure
