# Hessian spectral analysis and visualization functions

# This library is distributed under Apache 2.0 license

# (c) Kryptonite, 2024

import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import pickle

from src_lla.hessian import hessian_calc

# TODO: take this from config or allow user input with flag
default_viz_dir = 'viz_results'
default_res_dir = 'analysis_results'


def _coerce_matrix(data):
    try:
        arr = np.array(data, dtype=object)
    except Exception:
        return np.zeros((0, 0))

    if arr.size == 0:
        return np.zeros((0, 0))

    if arr.dtype == object:
        try:
            arr = [np.asarray(run, dtype=np.float64) for run in arr]
            arr = np.stack(arr, axis=0)
        except Exception:
            arr = np.zeros((0, 0))
    else:
        arr = arr.astype(np.float64)

    if arr.ndim == 1:
        arr = arr[np.newaxis, :]

    return arr


def _get_smoothing_fraction(default=0.005):
    env_val = os.getenv('LLA_ESD_SMOOTHING_FRAC')
    if env_val is not None:
        try:
            frac = float(env_val)
            if np.isfinite(frac) and frac > 0:
                return frac
        except Exception:
            pass
    return default


def _resolve_smoothing(span, override_s2=None):
    min_sigma = 1e-6
    if override_s2 is not None:
        try:
            variance = float(override_s2)
            if variance > 0:
                sigma = np.sqrt(variance)
                fraction = sigma / span if span > 0 else 0.0
                return variance, sigma, fraction
        except Exception:
            pass

    fraction = _get_smoothing_fraction()
    effective_span = span if span > 0 else 1.0
    sigma = max(effective_span * fraction, min_sigma)
    variance = sigma ** 2
    return variance, sigma, fraction


def hessian_criteria(eigenvalues,weights,n):
    
    """
    calculates Hessian criteria re Khn based on Gaussian quadrature
    of spectral decomposition, where n is a power

    :eigenvalues - eigenvalues of spectral decomposition
    :weights - weights of spectral decomposition    
    :n - power for eigenvalues*weights product, float; recommended 0<n<1
    returns re and Khn values (float)
    """

    if n < 0:
        warnings.warn('Using negative power for Khn calculation will lead to incrorrect results!')

    re = []
    Khn = []
    
    for i in range(len(eigenvalues)):
        eigs = np.array(eigenvalues[i])
    
        re.append(np.abs(np.min(eigs)/np.max(eigs)))
        
        eig_ws = np.real(np.array(weights[i]))
        eig_pos = np.sum(np.power(eigs[eigs>0]*eig_ws[eigs>0],n))
        eig_neg = np.sum(np.power(np.abs(eigs[eigs<0]*eig_ws[eigs<0]),n))
    
        Khn.append(np.abs(eig_neg/eig_pos))

    re = np.mean(np.array(re), axis=0)
    Khn = np.mean(np.array(Khn), axis=0)

    return re, Khn


def gaussian_conv(x, s2):
    return np.exp(-x**2 / (2.0 * s2)) / np.sqrt(2 * np.pi * s2)


def density_plot(eigenvalues, weights, num_bins=10000, s2=None, ext=0.02, return_metadata=False):
    """
    evaluates parameters of density plot in histogram form 

    :eigenvalues - eigenvalues of spectral decomposition
    :weights - weights of spectral decomposition
    :num_bins - number of eigenvalue bins in histogram
    :s2 - squared sigma used for gaussian convolution
    :ext - horizontal plot offset
    returns density and segments for plotting
    """

    eigenvalues = _coerce_matrix(eigenvalues)
    weights = _coerce_matrix(weights)

    if eigenvalues.size == 0 or weights.size == 0:
        segments = np.linspace(-1.0, 1.0, num=num_bins)
        density = np.zeros_like(segments)
        if return_metadata:
            return density, segments, {
                'smoothing_sigma': 0.0,
                'smoothing_variance': 0.0,
                'smoothing_fraction': 0.0,
            }
        return density, segments

    eig_min = float(np.min(eigenvalues))
    eig_max = float(np.max(eigenvalues))
    bound = max(abs(eig_min), abs(eig_max))
    if not np.isfinite(bound):
        bound = 1.0
    if bound == 0.0:
        bound = 1.0

    pad = float(ext) if ext is not None else 0.0
    if pad < 0:
        pad = 0.0
    segments = np.linspace(-bound - pad, bound + pad, num=num_bins)
    span = segments[-1] - segments[0]
    s2_effective, sigma, fraction = _resolve_smoothing(span, override_s2=s2)

    num_runs = eigenvalues.shape[0]
    bin_density = np.zeros((num_runs, num_bins))

    # calculating bin density values
    for i in range(num_runs):
        for j in range(num_bins):
            x = segments[j]
            bin_val = gaussian_conv(x - eigenvalues[i, :], s2_effective)
            bin_density[i, j] = np.sum(bin_val * weights[i, :])
    density = np.mean(bin_density, axis=0)
    density = density / (np.sum(density) * (segments[1] - segments[0])) # normalized
    if return_metadata:
        return density, segments, {
            'smoothing_sigma': float(sigma),
            'smoothing_variance': float(s2_effective),
            'smoothing_fraction': float(fraction),
        }
    return density, segments


def esd_plot(eigenvalues, weights,to_save=False,to_viz=True,exp_name='esd_example',viz_dir=default_viz_dir,to_return=False):

    """
    plots and saves esd in histogram form 

    :eigenvalues - eigenvalues of spectral decomposition
    :weights - weights of spectral decomposition
    :to_save - whether to save the results in viz_dir
    :to_viz - whether to show to plots (in notebook)
    :viz_dir - path to directory for output files
    :exp_name - tag of experiment used in names of output files
    :to_return whether to return density, segments (mainly for debugging)
    returns density, segments or none
    """

    if to_return:
        density_out = density_plot(eigenvalues, weights, return_metadata=True)
        density, segments, metadata = density_out
    else:
        density, segments = density_plot(eigenvalues, weights)
        metadata = None
    plt.semilogy(segments, density + 1.0e-7)
    plt.ylabel('Density (Log Scale)', fontsize=16, labelpad=10)
    plt.xlabel('Eigenvalues', fontsize=16, labelpad=10)
    plt.axis([segments[0], segments[-1], None, None])
    plt.tight_layout()

    if to_save:    
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        plt.savefig(os.path.join(viz_dir, exp_name + '_esd.png'))
        if not to_viz:
            plt.close()
    
    if to_viz:
        plt.show()
    else:
        plt.close()
        
    if to_return: # this option is for debug purposes only
        return density, segments, metadata


def eval_save_esd(hessian,n_iter=100,n_v=1,max_v=10,mask_idx=None,to_save=False,to_viz=True,exp_name='esd_example',
                  viz_dir=default_viz_dir,res_dir=default_res_dir,calc_crit=False,n_kh=0.5,return_data=False):

    """
    calculates and plots hessian esd

    :hessian - hessian class object
    :n_iter - max number of iterations for esd approximation
    :n_v - number of esd evaluation runs
    :max_v - max number of saved orthogonal vectors for esd approximation (increases required memory!!!)
    :mask_idx - list of layer indexes to keep for hessian calcualtions
    :to_save - whether to save the results in viz_dir
    :to_viz - whether to show to plots (in notebook)
    :viz_dir - path to directory for output files
    :exp_name - tag of experiment used in names of output files
    :return_data - if True, return ESD payload alongside criteria
    returns tuple re, Khn or None, None
    """

    eigs, weights = hessian.esd_calc(n_iter=n_iter,n_v=n_v,max_v=max_v,mask_idx=mask_idx)
    density = None
    segments = None
    metadata = None
    if return_data:
        density_out = esd_plot(
            eigs,
            weights,
            to_save=to_save,
            to_viz=to_viz,
            viz_dir=viz_dir,
            exp_name=exp_name,
            to_return=True,
        )
        if isinstance(density_out, tuple):
            if len(density_out) == 3:
                density, segments, metadata = density_out
            elif len(density_out) == 2:
                density, segments = density_out
        else:
            density = density_out
    else:
        esd_plot(eigs, weights, to_save=to_save,to_viz=to_viz,viz_dir=viz_dir,exp_name=exp_name)

    if calc_crit:
        re, Khn = hessian_criteria(eigs,weights,n_kh)

        if to_save:
            if not os.path.exists(res_dir):
                os.makedirs(res_dir)
            with open(os.path.join(res_dir,'hessian_criteria_{}.log'.format(exp_name)), 'a') as log_file:
                    log_file.write('re: {}, Kh{}: {}\n'.format(re,n_kh,Khn))

        if return_data:
            payload = {
                'eigenvalues_runs': eigs,
                'weights_runs': weights,
                'density': density if density is not None else np.array([]),
                'segments': segments if segments is not None else np.array([]),
            }
            if metadata:
                payload.update(metadata)
            return (re, Khn), payload
        return re, Khn

    if return_data:
        payload = {
            'eigenvalues_runs': eigs,
            'weights_runs': weights,
            'density': density if density is not None else np.array([]),
            'segments': segments if segments is not None else np.array([]),
        }
        if metadata:
            payload.update(metadata)
        return (None, None), payload

    return None, None


def viz_esd(model,metric,eigs=False,top_n=2,eigs_n_iter=100,eigs_tol=1e-3,trace=False,trace_n_iter=100,trace_tol=1e-3,
            esd=True,esd_n_iter=100,n_v=1,max_v=10,mask_idx=None,to_save=False, 
            to_viz=True,exp_name='esd_example',viz_dir=default_viz_dir,res_dir=default_res_dir,calc_crit=False,n_kh=0.5,
            return_data=False):

    """
    a funtions that collects different operations with hessian: eigs and esd

    :model - neural network torch model object
    :metric - loss evaluator Metric object
    :eigs - wheater to calculated hessian eigenvalues and eigenvectors or not
    :esd - whether to calculated hessian esd or not
    :calc_crit - whether to calculate hessian criteria re and Khn
    :n_kh - power for Khn criterion
    :top_n - number of top eigenvalues to compute
    :eigs_n_iter - number of iterations in eigs calculation
    :tol - tolerance to compare eigenvalues on consecutive iterations
    :esd_n_iter - max number of iterations for esd approximation
    :n_v - number of esd evaluation runs
    :max_v - max number of saved orthogonal vectors for esd approximation (increases required memory!!!)
    :mask_idx - list of layer indexes to keep for hessian calcualtions
    :to_save - whether to save the results (into viz_dir for plots and res_dir for criteria)
    :to_viz - whether to show to plots (in notebook)
    :viz_dir - path to directory to save output plots
    :res_dir - path to directory to save output results
    :exp_name - tag of experiment used in names of output files
    :return_data - whether to return raw ESD payload in addition to metrics
    returns a list of possible results [eigenvalues,eigenvectors,re,Khn] 
    """

    if calc_crit and not esd:
        raise AttributeError('Hessian criteria calculation is requested but esd calculation is not! Please call viz_esd with esd=True.')

    results = [None,None,None,None,None] # eigenvalues, eigenvectors, trace, re, Khn
    hessian = hessian_calc(model,metric)

    ### check if mask_idx is not list, or negative values in mask_idx, or values greater than the number of model layers are present
    if mask_idx is not None and (type(mask_idx) is not list or len([el for el in mask_idx if el<0])>0 or len([el for el in mask_idx if el>len(hessian.params)])>0):
        print('Warning! Invalid indexes encountered in mask index list, only positive numbers less than max model layer number are allowed!')
        print('Setting mask_idx to None')
        mask_idx = None
    esd_payload = None

    if esd:
        esd_result = eval_save_esd(
            hessian,
            n_iter=esd_n_iter,
            n_v=n_v,
            max_v=max_v,
            mask_idx=mask_idx,
            to_save=to_save,
            to_viz=to_viz,
            viz_dir=viz_dir,
            res_dir=res_dir,
            exp_name=exp_name,
            calc_crit=calc_crit,
            n_kh=n_kh,
            return_data=return_data,
        )
        if return_data:
            (re, Khn), esd_payload = esd_result
        else:
            re, Khn = esd_result

        if calc_crit: # this is redundant since eval_save_esd will return None, None if not calc_crit
            results[3] = re
            results[4] = Khn

    if eigs:
        res = hessian.eigs_calc(top_n=top_n,n_iter=eigs_n_iter,tol=eigs_tol,mask_idx=mask_idx)
        results[0] = res[0]
        results[1] = res[1]
        
    if trace:
        results[2] = hessian.tr_calc(n_iter=trace_n_iter,tol=trace_tol,mask_idx=mask_idx)
        
    hessian.reset()
    
    if to_save:
            
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        
        if results[0] is not None:
            with open(os.path.join(res_dir,'eigenvalues_{}.log'.format(exp_name)), 'a') as log_file:
                log_file.write('{}\n'.format(results[0]))
        if results[1] is not None:
            with open(os.path.join(res_dir,'eigenvectors_{}.pickle'.format(exp_name)), 'wb') as save_file:
                pickle.dump(results[1], save_file, protocol=pickle.HIGHEST_PROTOCOL)
        if results[2] is not None:
            with open(os.path.join(res_dir,'trace_{}.log'.format(exp_name)), 'a') as log_file:
                log_file.write('{}\n'.format(results[2]))

    if return_data:
        return results, esd_payload

    return results

