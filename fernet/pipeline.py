from __future__ import print_function
import numpy as np
from dipy.reconst import dti
from dipy.core.gradients import gradient_table_from_bvals_bvecs
from fernet.utils import erode_mask 
from fernet.free_water import grad_data_fit_tensor, clip_tensor_evals 

d = 3.0e-3

def estimate_tensor(dwi_data, mask, bvals, bvecs):
    '''
    Estimate the tensor image using dipy.
    '''
    gtab = gradient_table_from_bvals_bvecs(bvals, bvecs)
    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(dwi_data, mask=(mask > 0))
    tensor_data = tenfit.lower_triangular().astype('float32')
    tensor_data = tensor_data[...,np.newaxis,:] * mask[...,np.newaxis,np.newaxis]
    return tensor_data
    
def calculate_scalars(tensor_data, mask):
    '''
    Calculate the scalar images from the tensor
    returns: FA, MD, TR, AX, RAD
    '''
    mask = np.asarray(mask, dtype=np.bool)
    shape = mask.shape
    data = dti.from_lower_triangular(tensor_data[mask])
    w, v = dti.decompose_tensor(data)
    w = np.squeeze(w)
    v = np.squeeze(v)
    md = np.zeros(shape)
    md[mask] = dti.mean_diffusivity(w,axis=-1)
    fa = np.zeros(shape)
    fa[mask]  = dti.fractional_anisotropy(w,axis=-1)
    tr = np.zeros(shape)
    tr[mask]  = dti.trace(w,axis=-1)
    ax = np.zeros(shape)
    ax[mask]  = dti.axial_diffusivity(w,axis=-1)
    rad = np.zeros(shape)
    rad[mask]  = dti.radial_diffusivity(w,axis=-1)
    return fa, md, tr, ax, rad

def tissue_rois(mask, fa, tr, erode_iterations=10, fa_threshold=0.7, tr_threshold=0.0085, exclude=None):
    ''' 
    Calculate tissue ROIs inside a mask after eroding the mask
    With the option to exclude certain voxels
    '''
    mask = np.asarray(mask, dtype=np.bool)
    mask = erode_mask(mask, erode_iterations)
    if exclude is not None:
        mask = np.logical_and(mask, exclude==0)
    wm_roi = np.logical_and(mask, fa>fa_threshold)
    csf_roi = np.logical_and(mask, tr>tr_threshold)
    return wm_roi, csf_roi

def initial_fit(dwis, bvals, bvecs, mask, wm_roi, csf_roi, MD, 
        csf_percentile=95, wm_percentile=5, lmin=0.1e-3, lmax=2.5e-3, 
        evals_lmin=0.1e-3, evals_lmax=2.5e-3, md_value=0.6e-3, 
        interpolate=True, fixed_MD=False):
    '''
    Produce the initial estimate of the volume fraction and the initial tensor image
    '''
    print(" - Compute baseline image and DW attenuation.")
    dim_x, dim_y, dim_z = mask.shape
    indices_dwis = (bvals > 0)
    nb_dwis = np.count_nonzero(indices_dwis)
    indices_b0 = (bvals == 0)
    nb_b0 = np.count_nonzero(indices_b0)  
    # TO DO : address this line for multi-shell dwi 
    b = bvals.max()    
    b0 = dwis[..., indices_b0].mean(-1)
    # signal attenuation 
    signal = dwis[..., indices_dwis] / b0[..., None]
    np.clip(signal, 1.0e-6, 1-1.0e-6, signal)
    signal[np.logical_not(mask)] = 0.
    # tissue b0 references 
    csf_b0 = np.percentile(b0[csf_roi], csf_percentile) 
    print("\t{0:2d}th percentile of b0 signal in CSF: {1}.".format(csf_percentile, csf_b0))
    wm_b0 = np.percentile(b0[wm_roi], wm_percentile) 
    print("\t{0:2d}th percentile of b0 signal in WM : {1}.".format(wm_percentile, wm_b0))

    print(" - Compute initial volume fraction ...")
    # Eq. 7 from Pasternak 2009 MRM 
    epsi = 1e-12  # only used to prevent log(0)
    init_f = 1 - np.log(b0/wm_b0 + epsi)/np.log(csf_b0/wm_b0)
    np.clip(init_f, 0.0, 1.0, init_f)
    alpha = init_f.copy() # exponent for interpolation 
    
    print(" - Compute fixed MD VF map")
    init_f_MD = (np.exp(-b*MD)-np.exp(-b*d)) / (np.exp(-b*md_value)-np.exp(-b*d))
    np.clip(init_f_MD, 0.01, 0.99, init_f_MD)

    print(" - Compute min_f and max_f from lmin, lmax")
    ### This was following Pasternak 2009 although with an error 
    ### Amin = exp(-b*lmax)   and Amax = exp(-b*lmin)  in that paper 
    # min_f = (signal.min(-1)-np.exp(-b*d)) / (np.exp(-b*lmin)-np.exp(-b*d))
    # max_f = (signal.max(-1)-np.exp(-b*d)) / (np.exp(-b*lmax)-np.exp(-b*d))
    ### From Pasternak 2009 method, Amin < At implies that the 
    ### term with signal.min(-1) in numerator is the upper bound of f 
    ### although in that paper the equation 6 has fmin and fmax reversed. 
    ### With lmin, lmax=0.1e-3, 2.5e-3, Amin = 0.08, Awater = 0.04 
    ### and one can see that max_f here will usually be >> 1 
    min_f = (signal.max(-1)-np.exp(-b*d)) / (np.exp(-b*lmin)-np.exp(-b*d))
    max_f = (signal.min(-1)-np.exp(-b*d)) / (np.exp(-b*lmax)-np.exp(-b*d))
    # If MD of a voxel is > 3.0e-3, min_f and max_f can be negative. 
    # These voxels should be initialized as 0 
    np.clip(min_f, 0.0, 1.0, min_f)
    np.clip(max_f, 0.0, 1.0, max_f)
    np.clip(init_f, min_f, max_f, init_f)

    if interpolate:
        print(" - Interpolate two estimates of volume fraction") 
        # f = tissue fraction. with init_f high, alpha will be ~1 and init_f_MD will be weighted 
        init_f = (np.power(init_f, (1 - alpha))) * (np.power(init_f_MD, alpha))
    elif fixed_MD:
        print(" - Using fixed MD value of {0} for inital volume fraction".format(md_value))
        init_f = init_f_MD
    else:
        print(" - Using lmin and lmax for initial volume fraction") 
    
    np.clip(init_f, 0.05, 0.99, init_f)  # want minimum 5% of tissue
    init_f[np.isnan(init_f)] = 0.5
    init_f[np.logical_not(mask)] = 0.5
    
    print(" - Compute initial tissue tensor ...")
    signal[np.isnan(signal)] = 0
    bvecs = bvecs[indices_dwis]
    bvals = bvals[indices_dwis]
    signal_free_water = np.exp(-bvals * d)
    corrected_signal = (signal - (1 - init_f[..., np.newaxis]) \
                     * signal_free_water[np.newaxis, np.newaxis, np.newaxis, :]) \
                     / (init_f[..., np.newaxis])
    np.clip(corrected_signal, 1.0e-3, 1.-1.0e-3, corrected_signal)
    log_signal = np.log(corrected_signal)
    gtab = gradient_table_from_bvals_bvecs(bvals, bvecs)
    H = dti.design_matrix(gtab)[:, :6]
    pseudo_inv = np.dot(np.linalg.inv(np.dot(H.T, H)), H.T)
    init_tensor = np.dot(log_signal, pseudo_inv.T)

    dti_params = dti.eig_from_lo_tri(init_tensor).reshape((dim_x, dim_y, dim_z, 4, 
        3))
    evals = dti_params[..., 0, :]
    evecs = dti_params[..., 1:, :]
    if evals_lmin > 0.1e-3:
        print(" - Fatten tensor to {}".format(evals_lmin))
    lower_triangular = clip_tensor_evals(evals, evecs, evals_lmin, evals_lmax)
    lower_triangular[np.logical_not(mask)] = [evals_lmin, 0, evals_lmin, 0, 0, evals_lmin]
    nan_mask = np.any(np.isnan(lower_triangular), axis=-1)
    lower_triangular[nan_mask] = [evals_lmin, 0, evals_lmin, 0, 0, evals_lmin]

    init_tensor = lower_triangular[:, :, :, np.newaxis, :]
    return init_f, init_tensor

def gradient_descent(dwis, bvals, bvecs, mask, init_f, init_tensor, niters=50): 
    '''
    Optimize the volume fraction and the tensor via gradient descent.
    '''
    dim_x, dim_y, dim_z = mask.shape
    indices_dwis = (bvals > 0)
    nb_dwis = np.count_nonzero(indices_dwis)
    indices_b0 = (bvals == 0)
    nb_b0 = np.count_nonzero(indices_b0)  
    b = bvals.max()    
    b0 = dwis[..., indices_b0].mean(-1)
    signal = dwis[..., indices_dwis] / b0[..., None]
    np.clip(signal, 1.0e-6, 1-1.0e-6, signal)
    bvals = bvals[indices_dwis]
    bvecs = bvecs[indices_dwis]
    gtab = gradient_table_from_bvals_bvecs(bvals, bvecs)
    H = dti.design_matrix(gtab)[:, :6]
    signal[np.logical_not(mask)] = 0.
    signal = signal[mask]
    lower_triangular = init_tensor[mask, 0]
    volume_fraction = init_f[mask]
    print(" - Begin gradient descent.")
    mask_nvoxels = np.count_nonzero(mask)
    step_size = 1.0e-7
    weight = 100.0
    l_min_loop, l_max_loop = 0.1e-3, 2.5e-3
    
    for i in range(niters):
        print(" - Iteration {0:d} out of {1:d}.".format(i+1, niters))
        
        grad1, predicted_signal_tissue, predicted_signal_water = \
            grad_data_fit_tensor(lower_triangular, signal, H, bvals, 
                                 volume_fraction)
        print("\tgrad1 avg: {0:0.4e}".format(np.mean(np.abs(grad1))))
        predicted_signal = volume_fraction[..., None] * predicted_signal_tissue + \
                           (1-volume_fraction[..., None]) * predicted_signal_water
        prediction_error = np.sqrt(((predicted_signal - signal)**2).mean(-1))
        print("\tpref error avg: {0:0.4e}".format(np.mean(prediction_error)))
        
        gradf = (bvals * (predicted_signal - signal) \
                           * (predicted_signal_tissue - predicted_signal_water)).sum(-1)
        print("\tgradf avg: {0:0.4e}".format(np.mean(np.abs(gradf))))
        volume_fraction -= weight * step_size * gradf

        grad1[np.isnan(grad1)] = 0
        # np.clip(grad1, -1.e5, 1.e5, grad1)
        np.clip(volume_fraction, 0.01, 0.99, volume_fraction)
        lower_triangular -= step_size * grad1
        lower_triangular[np.isnan(lower_triangular)] = 0

        dti_params = dti.eig_from_lo_tri(lower_triangular).reshape((mask_nvoxels, 4, 
            3))
        evals = dti_params[..., 0, :]
        evecs = dti_params[..., 1:, :]
        lower_triangular = clip_tensor_evals(evals, evecs, l_min_loop, l_max_loop)
        del dti_params, evals, evecs

    final_tensor = np.zeros((dim_x, dim_y, dim_z, 1, 6), dtype=np.float32)
    final_tensor[mask, 0] = lower_triangular
    final_f = np.zeros((dim_x, dim_y, dim_z), dtype=np.float32)
    final_f[mask] = 1 - volume_fraction
    
    return final_f, final_tensor
