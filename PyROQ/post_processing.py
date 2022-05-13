import matplotlib, matplotlib.pyplot as plt, numpy as np, os, seaborn as sns

import linear_algebra

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
labels_fontsize = 16

## Functions to test the performance of the waveform representation, using the interpolant built from the selected basis.

def plot_representation_error(b, emp_nodes, paramspoint, term, outputdir, freq, paramspoint_to_wave):
    
    hp, hc = paramspoint_to_wave(paramspoint)
    
    if   term == 'lin':
        pass
    elif term == 'qua':
        hphc = np.real(hp * np.conj(hc))
        hp   = (np.absolute(hp))**2
        hc   = (np.absolute(hc))**2
    else              :
        raise TermError
    
    freq           = freq
    hp_emp, hc_emp = hp[emp_nodes], hc[emp_nodes]
    hp_rep, hc_rep = np.dot(b,hp_emp), np.dot(b,hc_emp)
    diff_hp        = hp_rep - hp
    diff_hc        = hc_rep - hc
    rep_error_hp   = diff_hp/np.sqrt(np.vdot(hp,hp))
    rep_error_hc   = diff_hc/np.sqrt(np.vdot(hc,hc))
    if term == 'qua':
        hphc_emp       = hphc[emp_nodes]
        hphc_rep       = np.dot(b,hphc_emp)
        diff_hphc      = hphc_rep - hphc
        rep_error_hphc = diff_hphc/np.sqrt(np.vdot(hphc,hphc))

    plt.figure(figsize=(8,5))
    if term == 'lin':
        plt.plot(    freq, np.real(hp),     color='orangered', lw=1.3, alpha=0.8, ls='-',  label='$\mathrm{Full}$')
        plt.plot(    freq, np.real(hp_rep), color='black',     lw=0.8, alpha=1.0, ls='--', label='$\mathrm{ROQ}$' )
    else:
        plt.semilogy(freq, np.real(hp),     color='orangered', lw=1.3, alpha=0.8, ls='-',  label='$mathrm{Full}$')
        plt.semilogy(freq, np.real(hp_rep), color='black',     lw=0.8, alpha=1.0, ls='--', label='$\mathrm{ROQ}$' )
    plt.scatter(freq[emp_nodes], np.real(hp)[emp_nodes], marker='o', c='dodgerblue', s=10, label='$\mathrm{Empirical \,\, nodes}$')
    plt.xlabel('$\mathrm{Frequency}$', fontsize=labels_fontsize)
    plt.ylabel('$\mathrm{\Re[h_+]}$', fontsize=labels_fontsize)
    plt.title('$\mathrm{Waveform \,\, comparison \,\, (%s \,\, basis)}$'%(term), fontsize=labels_fontsize)
    plt.legend(loc='best')
    plt.savefig(os.path.join(outputdir,'Plots/Waveform_comparison_hp_real_{}.pdf'.format(term)), bbox_inches='tight')

    plt.figure(figsize=(8,5))
    if term == 'lin':
        plt.plot(freq, np.real(hc),     color='orangered', lw=1.3, alpha=0.8, ls='-',  label='$\mathrm{Full}$')
        plt.plot(freq, np.real(hc_rep), color='black',     lw=0.8, alpha=1.0, ls='--', label='$\mathrm{ROQ}$' )
    else:
        plt.semilogy(freq, np.real(hc),     color='orangered', lw=1.3, alpha=0.8, ls='-',  label='$\mathrm{Full}$')
        plt.semilogy(freq, np.real(hc_rep), color='black',     lw=0.8, alpha=1.0, ls='--', label='$\mathrm{ROQ}$' )
    plt.scatter(freq[emp_nodes], np.real(hc)[emp_nodes], marker='o', c='dodgerblue', s=10, label='$\mathrm{Empirical \,\, nodes}$')
    plt.xlabel('$\mathrm{Frequency}$', fontsize=labels_fontsize)
    plt.ylabel('$\mathrm{\Re[h_{\\times}]}$', fontsize=labels_fontsize)
    plt.title('$\mathrm{Waveform \,\, comparison \,\, (%s \,\, basis)}$'%(term), fontsize=labels_fontsize)
    plt.legend(loc='best')
    plt.savefig(os.path.join(outputdir,'Plots/Waveform_comparison_hc_real_{}.pdf'.format(term)), bbox_inches='tight')

    if term == 'lin':
        plt.figure(figsize=(8,5))
        plt.plot(freq, np.imag(hp),     color='orangered', lw=1.3, alpha=0.8, ls='-',  label='$\mathrm{Full}$')
        plt.plot(freq, np.imag(hp_rep), color='black',     lw=0.8, alpha=1.0, ls='--', label='$\mathrm{ROQ}$' )
        plt.scatter(freq[emp_nodes], np.imag(hp)[emp_nodes], marker='o', c='dodgerblue', s=10, label='$\mathrm{Empirical \,\, nodes}$')
        plt.xlabel('$\mathrm{Frequency}$', fontsize=labels_fontsize)
        plt.ylabel('$\Im[h_+]$', fontsize=labels_fontsize)
        plt.title('$\mathrm{Waveform \,\, comparison \,\, (%s \,\, basis)}$'%(term), fontsize=labels_fontsize)
        plt.legend(loc='best')
        plt.savefig(os.path.join(outputdir,'Plots/Waveform_comparison_hp_imag_{}.pdf'.format(term)), bbox_inches='tight')

        plt.figure(figsize=(8,5))
        plt.plot(freq, np.imag(hc),     color='orangered', lw=1.3, alpha=0.8, ls='-',  label='$\mathrm{Full}$')
        plt.plot(freq, np.imag(hc_rep), color='black',     lw=0.8, alpha=1.0, ls='--', label='$\mathrm{ROQ}$' )
        plt.scatter(freq[emp_nodes], np.imag(hc)[emp_nodes], marker='o', c='dodgerblue', s=10, label='$\mathrm{Empirical \,\, nodes}$')
        plt.xlabel('$\mathrm{Frequency}$', fontsize=labels_fontsize)
        plt.ylabel('$\Im[h_{\\times}]$', fontsize=labels_fontsize)
        plt.title('$\mathrm{Waveform \,\, comparison \,\, (%s \,\, basis)}$'%(term), fontsize=labels_fontsize)
        plt.legend(loc='best')
        plt.savefig(os.path.join(outputdir,'Plots/Waveform_comparison_hc_imag_{}.pdf'.format(term)), bbox_inches='tight')

    else:
        plt.figure(figsize=(8,5))
        plt.plot(freq, hphc,     color='orangered', lw=1.3, alpha=0.8, ls='-',  label='$\mathrm{Full}$')
        plt.plot(freq, hphc_rep, color='black',     lw=0.8, alpha=1.0, ls='--', label='$\mathrm{ROQ}$' )
        plt.scatter(freq[emp_nodes], hphc[emp_nodes], marker='o', c='dodgerblue', s=10, label='$\mathrm{Empirical \,\, nodes}$')
        plt.xlabel('$\mathrm{Frequency}$', fontsize=labels_fontsize)
        plt.ylabel('$\Re[h_+ \, {h}^*_{\\times}]$', fontsize=labels_fontsize)
        plt.title('$\mathrm{Waveform \,\, comparison \,\, (%s \,\, basis)}$'%(term), fontsize=labels_fontsize)
        plt.legend(loc='best')
        plt.savefig(os.path.join(outputdir,'Plots/Waveform_comparison_hphc_real_{}.pdf'.format(term)), bbox_inches='tight')

        plt.figure(figsize=(8,5))
        plt.plot(   freq,            rep_error_hphc,            color='dodgerblue', lw=1.3, alpha=1.0, ls='-', label='$\Re[h_+ \, {h}^*_{\\times}]$')
        plt.scatter(freq[emp_nodes], rep_error_hphc[emp_nodes], color='dodgerblue', marker='o', s=10,          label='$\mathrm{Empirical \,\, nodes}$')
        plt.xlabel('$\mathrm{Frequency}$', fontsize=labels_fontsize)
        plt.ylabel('$\mathrm{Fractional Representation Error}$', fontsize=labels_fontsize)
        plt.title('$\mathrm{Representation \,\, Error \,\, (%s \,\, basis)}$'%(term), fontsize=labels_fontsize)
        plt.legend(loc='best')
        plt.savefig(os.path.join(outputdir,'Plots/Representation_error_hp_{}.pdf'.format(term)), bbox_inches='tight')

    plt.figure(figsize=(8,5))
    plt.plot(freq, np.real(rep_error_hp), color='dodgerblue', lw=1.3, alpha=1.0, ls='-', label='$\Re[h_+]$')
    if term == 'lin':
        plt.plot(freq, np.imag(rep_error_hp), color='darkred',    lw=1.3, alpha=0.8, ls='-', label='$\Im[h_+]$')
    plt.scatter(freq[emp_nodes], np.real(rep_error_hp)[emp_nodes], marker='o', c='dodgerblue', s=10, label='$\mathrm{Empirical \,\, nodes}$')
    plt.scatter(freq[emp_nodes], np.imag(rep_error_hp)[emp_nodes], marker='o', c='dodgerblue', s=10)
    plt.xlabel('$\mathrm{Frequency}$', fontsize=labels_fontsize)
    plt.ylabel('$\mathrm{Fractional \,\, Representation \,\, Error}$', fontsize=labels_fontsize)
    plt.title('$\mathrm{Representation \,\, Error \,\, (%s \,\, basis)}$'%(term), fontsize=labels_fontsize)
    plt.legend(loc='best')
    plt.savefig(os.path.join(outputdir,'Plots/Representation_error_hp_{}.pdf'.format(term)), bbox_inches='tight')

    plt.figure(figsize=(8,5))
    plt.plot(freq, np.real(rep_error_hc), color='dodgerblue', lw=1.3, alpha=1.0, ls='-', label='$\Re[h_{\\times}]$')
    if term == 'lin':
        plt.plot(freq, np.imag(rep_error_hc), color='darkred',    lw=1.3, alpha=0.8, ls='-', label='$\Im[h_{\\times}]$')
    plt.scatter(freq[emp_nodes], np.real(rep_error_hc)[emp_nodes], marker='o', c='dodgerblue', s=10, label='$\mathrm{Empirical \,\, nodes}$')
    plt.scatter(freq[emp_nodes], np.imag(rep_error_hc)[emp_nodes], marker='o', c='dodgerblue', s=10)
    plt.xlabel('$\mathrm{Frequency}$', fontsize=labels_fontsize)
    plt.ylabel('$\mathrm{Fractional \,\, Representation \,\, Error}$', fontsize=labels_fontsize)
    plt.title('$\mathrm{Representation \,\, Error \,\, (%s \,\, basis)}$'%(term), fontsize=labels_fontsize)
    plt.legend(loc='best')
    plt.savefig(os.path.join(outputdir,'Plots/Representation_error_hc_{}.pdf'.format(term)), bbox_inches='tight')

    return

def test_roq_error(b, emp_nodes, term, pyroq):
    
    # Initialise structures
    nsamples  = pyroq.n_tests_post
    ndim      = len(emp_nodes)
    surros_hp = np.zeros(nsamples)
    surros_hc = np.zeros(nsamples)
    
    # Draw random test points
    paramspoints = pyroq.generate_params_points(npts=nsamples)
    
    # Select tolerance
    if   term == 'lin':
        tol = pyroq.tolerance_lin
        pass
    elif term == 'qua':
        surros_hphc = np.zeros(nsamples)
        tol = pyroq.tolerance_qua
    else:
        raise TermError
    
    # Start looping over test points
    print('\n\n###########################################\n# Starting surrogate tests {} iteration #\n###########################################\n'.format(term))
    print('Tolerance: ', tol)
    for i,paramspoint in enumerate(paramspoints):
        
        # Generate test waveform
        hp, hc = pyroq.paramspoint_to_wave(paramspoint)
        
        # Compute quadratic terms and interpolant representations
        if term == 'qua':
            hphc     = np.real(hp * np.conj(hc))
            hphc_emp = hphc[emp_nodes]
            hphc_rep = np.dot(b,hphc_emp)
        
            hp       = (np.absolute(hp))**2
            hc       = (np.absolute(hc))**2

        hp_emp    = hp[emp_nodes]
        hp_rep    = np.dot(b,hp_emp)
        hc_emp    = hc[emp_nodes]
        hc_rep    = np.dot(b,hc_emp)

        # Compute the representation error. This is the same measure employed to stop adding elements to the basis
        surros_hp[i] = linear_algebra.overlap_of_two_waveforms(hp, hp_rep, pyroq.deltaF, pyroq.error_version)
        surros_hc[i] = linear_algebra.overlap_of_two_waveforms(hc, hc_rep, pyroq.deltaF, pyroq.error_version)
        if term == 'qua':
            surros_hphc[i] = linear_algebra.overlap_of_two_waveforms(hphc, hphc_rep, pyroq.deltaF, pyroq.error_version)

        # If a test case exceeds the error, let the user know. Always print typical test result every 100 steps
        np.set_printoptions(suppress=True)
        if pyroq.verbose:
            if (surros_hp[i] > tol): print("h_+     above tolerance: Iter: ", i, "Surrogate value: ", surros_hp[i], "Parameters: ", paramspoints[i])
            if (surros_hc[i] > tol): print("h_x     above tolerance: Iter: ", i, "Surrogate value: ", surros_hc[i], "Parameters: ", paramspoints[i])
#                if ((term == 'qua') and (surros_hphc[i] > tol)):
#                    print("h_+ h_x above tolerance: Iter: ", i, "Surrogate value: ", surros_hphc[i], "Parameters: ", paramspoints[i])
            if i%100==0:
                print("h_+     rolling check (every 100 steps): Iter: ",             i, "Surrogate value: ", surros_hp[i])
                print("h_x     rolling check (every 100 steps): Iter: ",             i, "Surrogate value: ", surros_hc[i])
#                    if (term == 'qua'):
#                        print("h_+ h_x rolling check (every 100 steps): Iter: ",             i, "Surrogate value: ", surros_hphc[i])
        np.set_printoptions(suppress=False)

    # Plot the test results
    plt.figure(figsize=(8,5))
    plt.semilogy(surros_hp, 'x', color='darkred',    label='$\Re[h_+]$')
#        plt.semilogy(surros_hc, 'x', color='dodgerblue', label='$\Re[h_{\\times}]$')
#        if term == 'qua':
#            plt.semilogy(surros_hphc,'o', label='h_+ * conj(h_x)')
    plt.xlabel('$\mathrm{Number \,\, of \,\, Random \,\, Test \,\, Points}$', fontsize=labels_fontsize)
    plt.ylabel('$\mathrm{Surrogate \,\, Error \,\, (%s \,\, basis)}$'%(term), fontsize=labels_fontsize)
    plt.legend(loc='best')
    plt.savefig(os.path.join(pyroq.outputdir,'Plots/Surrogate_errors_random_test_points_{}.pdf'.format(term)), bbox_inches='tight')

    return

def histogram_basis_params(params_basis, outputdir, i2n):

    p = {}
    for i,k in i2n.items():
        p[k] = []
        for j in range(len(params_basis)):
            p[k].append(params_basis[j][i])
        
        plt.figure()
        sns.displot(p[k], color='darkred')
        plt.xlabel(k, fontsize=labels_fontsize)
        plt.savefig(os.path.join(outputdir,'Plots/Basis_parameters_{}.pdf'.format(k)), bbox_inches='tight')
        plt.close()
