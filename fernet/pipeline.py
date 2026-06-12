import os, sys
import logging
import warnings
import numpy as np
import nibabel as nib
from pathlib import Path
FERNET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FERNET_ROOT))
from dipy.reconst import dti
from dipy.core.gradients import gradient_table_from_bvals_bvecs
from helper import read_dwi, read_nifti, write_dwi, nifti_image
from helper import n4_bias_correct_dwi, has_gaussian_shells, extract_gaussian_shells, round_bvals 
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
    tensor_data = tenfit.lower_triangular().astype(np.float32)
    tensor_data = tensor_data[...,np.newaxis,:] * mask[...,np.newaxis,np.newaxis]
    return tensor_data
    
def calculate_scalars(tensor_data, mask):
    '''
    Calculate the scalar images from the tensor
    returns: FA, MD, TR, AX, RAD
    '''
    mask = np.asarray(mask, dtype=bool)
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
    mask = np.asarray(mask, dtype=bool)
    mask = erode_mask(mask, erode_iterations)
    if exclude is not None:
        mask = np.logical_and(mask, exclude==0)
    wm_roi = np.logical_and(mask, fa>fa_threshold)
    csf_roi = np.logical_and(mask, tr>tr_threshold)
    return wm_roi, csf_roi

def gradient_table_no_bzero(bvals, bvecs):
    '''
    Runs dipy gradient_table_from_bvals_bvecs while suppressing warnings about b0_threshold
    '''
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*b0_threshold.*is too low.*",
            category=UserWarning,
            module=r"dipy\.core\.gradients"
        )
        gtab = gradient_table_from_bvals_bvecs(bvals, bvecs)
    return gtab 


def initial_fit(dwi, bvals, bvecs, mask, wm_roi, csf_roi, MD, 
        csf_percentile=95, wm_percentile=5, lmin=0.1e-3, lmax=2.5e-3, 
        evals_lmin=0.1e-3, evals_lmax=2.5e-3, md_value=0.6e-3, 
        interpolate=True, fixed_MD=False):
    '''
    Produce the initial estimate of the volume fraction and the initial tensor image
    '''
    logging.info("Compute baseline image and DW attenuation.")
    dim_x, dim_y, dim_z = mask.shape
    indices_dwi = (bvals > 0)
    indices_b0 = (bvals == 0)
    b = bvals.max()    
    b0 = dwi[..., indices_b0].mean(-1)
    # suppress expected divide by zero warning 
    with np.errstate(divide='ignore', invalid='ignore'):
        signal = dwi[..., indices_dwi] / b0[..., None]
    np.clip(signal, 1.0e-6, 1-1.0e-6, signal)
    signal[np.logical_not(mask)] = 0.
    csf_b0 = np.percentile(b0[csf_roi], csf_percentile) 
    logging.debug("{0:2d}th percentile of b0 signal in CSF: {1}.".format(csf_percentile, csf_b0))
    wm_b0 = np.percentile(b0[wm_roi], wm_percentile) 
    logging.debug("{0:2d}th percentile of b0 signal in WM : {1}.".format(wm_percentile, wm_b0))

    logging.info("Compute initial volume fraction ..." )
    epsi = 1e-12 # to prevent log(0)
    # suppress expected invalid value in log warning 
    with np.errstate(invalid='ignore'):
        init_f = 1 - np.log(b0/wm_b0 + epsi)/np.log(csf_b0/wm_b0)
    np.clip(init_f, 0.00, 1.00, init_f)
    alpha = init_f.copy()

    logging.info("Compute fixed MD VF map")
    init_f_MD = (np.exp(-b*MD)-np.exp(-b*d)) / (np.exp(-b*md_value)-np.exp(-b*d))
    np.clip(init_f_MD, 0.01, 0.99, init_f_MD)

    logging.info("Compute min_f and max_f from lmin, lmax")
    # This is Pasternak 2009 paper as written with typo
    # min_f = (signal.min(-1)-np.exp(-b*d)) / (np.exp(-b*lmax)-np.exp(-b*d))
    # max_f = (signal.max(-1)-np.exp(-b*d)) / (np.exp(-b*lmin)-np.exp(-b*d))
    # This is from original diffusion_manu code with bug 
    # min_f = (signal.min(-1)-np.exp(-b*d)) / (np.exp(-b*lmin)-np.exp(-b*d))
    # max_f = (signal.max(-1)-np.exp(-b*d)) / (np.exp(-b*lmax)-np.exp(-b*d))
    # Corrected below 
    min_f = (signal.max(-1)-np.exp(-b*d)) / (np.exp(-b*lmin)-np.exp(-b*d))
    max_f = (signal.min(-1)-np.exp(-b*d)) / (np.exp(-b*lmax)-np.exp(-b*d))
    np.clip(min_f, 0.0, 1.0, min_f)
    np.clip(max_f, 0.0, 1.0, max_f)
    np.clip(init_f, min_f, max_f, init_f)

    if interpolate:
        logging.info("Interpolate 2 estimates of volume fraction" )
        init_f = (np.power(init_f,(1-alpha)))*(np.power(init_f_MD,alpha))
    elif fixed_MD:
        logging.info("Using fixed MD value of {0} for inital volume fraction".format(md_value))
        init_f = init_f_MD
    else:
        logging.info("Using lmin and lmax for initial volume fraction" )
    
    np.clip(init_f, 0.05, 0.99, init_f)  # want minimum 5% of tissue
    init_f[np.isnan(init_f)] = 0.5
    init_f[np.logical_not(mask)] = 0.5
    
    logging.info("Compute initial tissue tensor ...")
    signal[np.isnan(signal)] = 0
    bvecs = bvecs[indices_dwi]
    bvals = bvals[indices_dwi]
    signal_free_water = np.exp(-bvals * d)
    corrected_signal = (signal - (1 - init_f[..., np.newaxis]) \
                     * signal_free_water[np.newaxis, np.newaxis, np.newaxis, :]) \
                     / (init_f[..., np.newaxis])
    np.clip(corrected_signal, 1.0e-3, 1.-1.0e-3, corrected_signal)
    log_signal = np.log(corrected_signal)
    gtab = gradient_table_no_bzero(bvals, bvecs)
    H = dti.design_matrix(gtab)[:, :6]
    pseudo_inv = np.dot(np.linalg.inv(np.dot(H.T, H)), H.T)
    init_tensor = np.dot(log_signal, pseudo_inv.T)

    dti_params = dti.eig_from_lo_tri(init_tensor).reshape((dim_x, dim_y, dim_z, 4, 
        3))
    evals = dti_params[..., 0, :]
    evecs = dti_params[..., 1:, :]
    if evals_lmin > 0.1e-3:
        logging.info("Fatten tensor to {}".format(evals_lmin))
    lower_triangular = clip_tensor_evals(evals, evecs, evals_lmin, evals_lmax)
    lower_triangular[np.logical_not(mask)] = [evals_lmin, 0, evals_lmin, 0, 0, evals_lmin]
    nan_mask = np.any(np.isnan(lower_triangular), axis=-1)
    lower_triangular[nan_mask] = [evals_lmin, 0, evals_lmin, 0, 0, evals_lmin]

    init_tensor = lower_triangular[:, :, :, np.newaxis, :]
    return init_f, init_tensor

def gradient_descent(dwi, bvals, bvecs, mask, init_f, init_tensor, niters=50, weight=100.0, step_size=1.0e-7): 
    '''
    Optimize the volume fraction and the tensor via gradient descent.
    '''
    dim_x, dim_y, dim_z = mask.shape
    indices_dwi = (bvals > 0)
    indices_b0 = (bvals == 0)
    b0 = dwi[..., indices_b0].mean(-1)
    # suppress expected divide by zero warning 
    with np.errstate(divide='ignore', invalid='ignore'):
        signal = dwi[..., indices_dwi] / b0[..., None]
    np.clip(signal, 1.0e-6, 1-1.0e-6, signal)
    bvals = bvals[indices_dwi]
    bvecs = bvecs[indices_dwi]
    gtab = gradient_table_no_bzero(bvals, bvecs)
    H = dti.design_matrix(gtab)[:, :6]
    signal[np.logical_not(mask)] = 0.
    signal = signal[mask]
    lower_triangular = init_tensor[mask, 0]
    volume_fraction = init_f[mask]
    logging.info("Begin gradient descent.")
    mask_nvoxels = np.count_nonzero(mask)
    l_min_loop, l_max_loop = 0.1e-3, 2.5e-3
    for i in range(niters):
        logging.info("Iteration {} out of {}.".format((i + 1), niters))
        
        grad1, predicted_signal_tissue, predicted_signal_water = \
            grad_data_fit_tensor(lower_triangular, signal, H, bvals, 
                                 volume_fraction)
        logging.debug("grad1 avg: {}".format(np.mean(np.abs(grad1))))
        predicted_signal = volume_fraction[..., None] * predicted_signal_tissue + \
                           (1-volume_fraction[..., None]) * predicted_signal_water
        prediction_error = np.sqrt(((predicted_signal - signal)**2).mean(-1))
        logging.debug("pref error avg: {}".format(np.mean(prediction_error)))
        
        gradf = (bvals * (predicted_signal - signal) \
                           * (predicted_signal_tissue - predicted_signal_water)).sum(-1)
        logging.debug("gradf avg: {}".format(np.mean(np.abs(gradf))))
        volume_fraction -= weight * step_size * gradf

        grad1[np.isnan(grad1)] = 0
        np.clip(volume_fraction, 0.01, 0.99, volume_fraction)
        lower_triangular -= step_size * grad1
        lower_triangular[np.isnan(lower_triangular)] = 0

        dti_params = dti.eig_from_lo_tri(lower_triangular).reshape((mask_nvoxels, 4, 
            3))
        evals = dti_params[..., 0, :]
        evecs = dti_params[..., 1:, :]
        lower_triangular = clip_tensor_evals(evals, evecs, l_min_loop, l_max_loop)
        del dti_params, evals, evecs

    final_tensor = np.zeros((dim_x, dim_y, dim_z, 1, 6), dtype=float)
    final_tensor[mask, 0] = lower_triangular
    final_f = np.zeros((dim_x, dim_y, dim_z), dtype=np.float32)
    final_f[mask] = 1 - volume_fraction
    
    return final_f, final_tensor
    
def run_fernet(dwi_filename, bvals_filename, bvecs_filename, mask_filename, output_basename,
                wm_roi=None, csf_roi=None, exclude_mask=None, interpolate=True, fixed_MD=False, 
                erode_iterations=8, fa_threshold=0.7, tr_threshold=0.0085, wm_percentile=5, csf_percentile=95, 
                md_value=0.6e-3, lmin=0.1e-3, lmax=2.5e-3, evals_lmin=0.1e-3, evals_lmax=2.5e-3, 
                niters=50, weight=100.0, step_size=1.0e-7, gaussian=True, bias_correct=False
                ):
    output_basename = os.path.realpath(output_basename)
    
    logging.info("Read DWIs from disk...")
    dwi_img, bvals, bvecs = read_dwi(dwi_filename, bvals_filename, bvecs_filename)
    bvals = round_bvals(bvals)
    logging.info("Read mask image from disk...")
    mask_img = read_nifti(mask_filename)
    mask = np.asarray(mask_img.get_fdata(), dtype=bool)
    if mask.sum() == 0:
        raise ValueError("Brain mask has zero volume")
    if wm_roi is not None:
        logging.info("Read provided WM ROI")
        wm_roi = np.asarray(read_nifti(wm_roi).get_fdata(), dtype=bool)    
        if wm_roi.sum() == 0:
            raise ValueError("Provided WM ROI has zero volume")
    if csf_roi is not None:
        logging.info("Read provided CSF ROI")
        csf_roi = np.asarray(read_nifti(csf_roi).get_fdata(), dtype=bool)    
        if csf_roi.sum() == 0:
            raise ValueError("Provided CSF ROI has zero volume")
    if gaussian and not has_gaussian_shells(bvals):
        raise ValueError("FERNET cannot be run on datasets without b-values in the range 500-1500 s/mm2")
    if gaussian:
        dwi_img, bvals, bvecs = extract_gaussian_shells(dwi_img, bvals, bvecs)
    if bias_correct:
        dwi_img, bvals, bvecs = n4_bias_correct_dwi(dwi_img, bvals, bvecs, mask_img=mask_img)
    # Begin 
    bvals = bvals.flatten()
    bvecs = bvecs.transpose()
    affine = dwi_img.affine
    dwi = dwi_img.get_fdata()
    dwi[np.logical_not(mask),...] = 0 
    
    logging.info("First, fit a standard tensor and calculate FA and MD" )
    tensor_data = estimate_tensor(dwi, mask, bvals, bvecs)
    FA, MD, TR, AX, RAD = calculate_scalars(tensor_data, mask)

    if (wm_roi is None) and (csf_roi is None):
        logging.info("Determine CSF and WM rois.")
        if exclude_mask:
            logging.info("Exclude voxels in exclude mask .")
            exclude_mask = np.asarray(read_nifti(exclude_mask).get_fdata(), dtype=bool)
        wm_roi, csf_roi = tissue_rois(mask, FA, TR, 
            erode_iterations=erode_iterations, 
            fa_threshold=fa_threshold, 
            tr_threshold=tr_threshold, 
            exclude=exclude_mask)
        
    init_f, init_tensor = initial_fit(dwi, bvals, bvecs, mask, wm_roi, csf_roi, MD, 
            csf_percentile=csf_percentile, wm_percentile=wm_percentile, 
            lmin=lmin, lmax=lmax, 
            evals_lmin=evals_lmin, evals_lmax=evals_lmax, md_value=md_value, 
            interpolate=interpolate, fixed_MD=fixed_MD)
        
    logging.info("Save initial volume fraction image as Nifti")
    init_f_img = nifti_image(init_f, affine, cal_max=1)
    nib.save(init_f_img, '{0}_init_volume_fraction.nii.gz'.format(output_basename))

    logging.info("Save initial tensor image as Nifti" )
    init_tensor_img = nifti_image(init_tensor, affine, intent_code=1005)
    nib.save(init_tensor_img, '{0}_init_tensor.nii.gz'.format(output_basename))

    logging.info("Begin gradient descent.")
    final_f, final_tensor = gradient_descent(dwi, bvals, bvecs, mask, 
            init_f, init_tensor, niters=niters, weight=weight, step_size=step_size)
    
    logging.info("Save tensor image as Nifti.")
    final_tensor_img = nifti_image(final_tensor, affine, intent_code=1005)
    nib.save(final_tensor_img, "{}_fw_tensor.nii.gz".format(output_basename))

    logging.info("Save volume fraction image as Nifti")
    final_f_img = nifti_image(final_f, affine, cal_max=1)
    nib.save(final_f_img, "{}_fw_volume_fraction.nii.gz".format(output_basename))

    logging.info("Save Fernet tensor scalars FA, TR, AX, RAD, FA difference as Nifti." )
    fw_fa, fw_md, fw_tr, fw_ax, fw_rad = calculate_scalars(final_tensor, mask)
    fw_fa_img = nifti_image(fw_fa, affine, cal_max=1)
    fw_tr_img = nifti_image(fw_tr, affine)
    fw_ax_img = nifti_image(fw_ax, affine)
    fw_rad_img = nifti_image(fw_rad, affine)
    fw_diff_fa_img = nifti_image(fw_fa - FA, affine)
    nib.save(fw_fa_img, "{}_fw_tensor_FA.nii.gz".format(output_basename))
    nib.save(fw_tr_img, "{}_fw_tensor_TR.nii.gz".format(output_basename))
    nib.save(fw_ax_img, "{}_fw_tensor_AX.nii.gz".format(output_basename))
    nib.save(fw_rad_img, "{}_fw_tensor_RAD.nii.gz".format(output_basename))
    nib.save(fw_diff_fa_img, "{}_difference_FA.nii.gz".format(output_basename))
    
    logging.info("Save WM and CSF rois as Nifti." )
    csf_roi_img = nifti_image(csf_roi.astype(np.int32), affine)
    wm_roi_img = nifti_image(wm_roi.astype(np.int32), affine)
    nib.save(csf_roi_img, "{}_csf_mask.nii.gz".format(output_basename))
    nib.save(wm_roi_img, "{}_wm_mask.nii.gz".format(output_basename))
    
def fernet_correct_dwi(dwi_filename, bvals_filename, bvecs_filename, mask_filename, volume_fraction_filename, output_basename):
    output_basename = os.path.realpath(output_basename)
    logging.info("Read DWIs from disk...")
    dwi_img, bvals, bvecs = read_dwi(dwi_filename, bvals_filename, bvecs_filename)
    bvals = bvals.flatten()
    bvecs = bvecs.transpose()
    affine = dwi_img.affine
    dwi = dwi_img.get_fdata()
    logging.info("Read mask image from disk...")
    mask_img = read_nifti(mask_filename)
    mask = np.asarray(mask_img.get_fdata(), dtype=bool)
    dwi[np.logical_not(mask),...] = 0 
    logging.info("Read volume fraction image from disk...")
    vf_img = read_nifti(volume_fraction_filename)
    fw_vf = vf_img.get_fdata()
    b0 = np.mean(dwi[...,bvals==0], axis=-1)
    tissue_vf = 1 - fw_vf
    b0_corrected = b0 * tissue_vf 
    atten = dwi / b0[...,np.newaxis] 
    atten_tissue = (atten - (fw_vf[...,np.newaxis] * np.exp(-1*bvals*d)[np.newaxis,np.newaxis,np.newaxis,...])) / tissue_vf[...,np.newaxis]
    atten_tissue[np.isnan(atten_tissue)] = 0
    b0_corrected[np.isnan(b0_corrected)] = 0 
    dwi_corrected  = atten_tissue * b0_corrected[...,np.newaxis]
    dwi_corrected[np.isnan(dwi_corrected)] = 0 
    dwi_corrected = np.clip(dwi_corrected, 0, None)
    dwi_corrected[fw_vf > 0.95,...] = 0 # we don't trust the tissue compartment if water is greater than 95 % 
    dwi_corrected_img = nifti_image(dwi_corrected, affine)
    logging.info("Write free water corrected DWI " )
    write_dwi(output_basename+'_fw_DWI.nii.gz', dwi_corrected_img, bvals[None,:], bvecs.T)
    logging.info("Write free water corrected B0 " )
    b0_corrected_img = nifti_image(b0_corrected, affine)
    b0_corrected_img.to_filename(output_basename+'_fw_B0.nii.gz')
    
def fernet_regions(dwi_filename, bvals_filename, bvecs_filename, mask_filename, output_basename,
                exclude_mask=None, fa_threshold=0.7, tr_threshold=0.0085, erode_iterations=8):
    output_basename = os.path.realpath(output_basename)
    logging.info("Read DWIs from disk...")
    dwi_img, bvals, bvecs = read_dwi(dwi_filename, bvals_filename, bvecs_filename)
    bvals = bvals.flatten()
    bvecs = bvecs.transpose()
    affine = dwi_img.affine
    dwi = dwi_img.get_fdata()
    logging.info("Read mask image from disk...")
    mask_img = read_nifti(mask_filename)
    mask = np.asarray(mask_img.get_fdata(), dtype=bool)
    dwi[np.logical_not(mask),...] = 0 
    logging.info("First, fit a standard tensor and calculate FA and MD" )
    tensor_data = estimate_tensor(dwi, mask, bvals, bvecs)
    FA, MD, TR, AX, RAD = calculate_scalars(tensor_data, mask)
    if exclude_mask:
        logging.info("Exclude voxels in exclude mask .")
        exclude_mask = np.asarray(read_nifti(exclude_mask).get_fdata(), dtype=bool)
    logging.info("Erode mask {} times".format(erode_iterations))
    logging.info("Threshold FA at {} and TR at {}".format(fa_threshold, tr_threshold))
    wm_roi, csf_roi = tissue_rois(mask, FA, TR, 
        erode_iterations=erode_iterations, fa_threshold=fa_threshold, 
        tr_threshold=tr_threshold, exclude=exclude_mask)
    wm_roi_img = nifti_image(wm_roi.astype(np.int32), affine)
    csf_roi_img = nifti_image(csf_roi.astype(np.int32), affine)
    logging.info("Save ROIs " )
    nib.save(wm_roi_img, "{}_wm_mask.nii.gz".format(output_basename))
    nib.save(csf_roi_img, "{}_csf_mask.nii.gz".format(output_basename))
