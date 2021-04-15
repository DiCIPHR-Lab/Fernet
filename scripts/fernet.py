#! /usr/bin/env python 
from __future__ import print_function
import os
import numpy as np
import nibabel as nib
import argparse
from fernet.pipeline import estimate_tensor, calculate_scalars, tissue_rois, initial_fit, gradient_descent
from fernet.utils import nifti_image, read_dwi

DESCRIPTION = '''
FERNET : FreewatER iNvariant Estimation of Tensor
Initialize the volume fraction map and free-water corrected tensor.
'''

# FERNET parameters
erode_iterations = 8
fa_threshold = 0.7
tr_threshold = 0.0085
md_value = 0.6e-3
lmax = 2.5e-3
lmin = 0.1e-3  
evals_lmin = 0.1e-3 
evals_lmax = 2.5e-3 
wm_percentile, csf_percentile = 5, 95
interpolate=True
fixed_MD=False

def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument('-d','-k','--data',action='store',metavar='data',dest='dwis_filename',
                    type=str, required=True, 
                    help='Input DWIs data file (Nifti or Analyze format).'
                    )
    p.add_argument('-r','--bvecs', action='store', metavar='bvecs', dest='bvecs',
                    type=str, required=False, default=None,
                    help='Gradient directions (.bvec file).'
                    )
    p.add_argument('-b','--bvals', action='store', metavar='bvals', dest='bvals',
                    type=str, required=False, default=None,
                    help='B-values (.bval file).'
                    )
    p.add_argument('-m','--mask',action='store',metavar='mask', dest='mask',
                    type=str, required=True, 
                    help='Brain mask file (Nifti or Analyze format).'
                    )
    p.add_argument('-w','--wm',action='store',metavar='wm',dest='wm',
                    type=str, required=False, default=None, 
                    help='An ROI defined in deep WM, e.g. corpus callosum (Nifti or Analyze format).'
                    )
    p.add_argument('-c','--csf',action='store',metavar='csf',dest='csf',
                    type=str, required=False, default=None, 
                    help='An ROI defined in CSF, e.g. ventricle (Nifti or Analyze format).'
                    )
    p.add_argument('-x','--exclude',action='store',metavar='mask',dest='exclude_mask',
                    type=str, required=False, default=None, 
                    help='A mask (e.g. peritumoral region) of voxels to exclude when getting typical WM, GM voxels.'
                    )
    p.add_argument('-o', '--output', action='store', metavar='output', dest='output',
                    type=str, required=True,
                    help='Output basename for init tensor map and volume fraction.'
                    )
    p.add_argument('-n', '--niters', action='store', metavar='niters', dest='niters', 
                    type=int, required=False, default=50,
                    help='Number of iterations of the gradient descent. Default is 50'
                    )
    return p

def main():    
    parser = buildArgsParser()
    args = parser.parse_args()
    dwis_filename = args.dwis_filename
    mask_filename = args.mask
    bvals_filename = args.bvals
    bvecs_filename = args.bvecs
    exclude_mask = args.exclude_mask 
    wm_roi = args.wm
    csf_roi = args.csf
    output_basename = args.output
    niters = args.niters
    print('''
    -------------------------------------------------------
       ________) _____) _____    __     __) _____) ______)
      (, /     /       (, /   ) (, /|  /  /       (, /
        /___,  )__       /__ /    / | /   )__       /
     ) /     /        ) /   \_ ) /  |/  /        ) /
    (_/     (_____)  (_/      (_/   '  (_____)  (_/

     FreewatER EstimatoR using iNtErpolated iniTialization
    -------------------------------------------------------

''')
    run_fernet(dwis_filename, bvals_filename, bvecs_filename, mask_filename, output_basename,
                wm_roi=wm_roi, csf_roi=csf_roi, exclude_mask=exclude_mask, niters=niters)
                
def run_fernet(dwis_filename, bvals_filename, bvecs_filename, mask_filename, output_basename,
                wm_roi=None, csf_roi=None, exclude_mask=None, niters=50):
    
    print(" - Read DWIs from disk...")
    dwis_img, bvals, bvecs = read_dwi(dwis_filename, bvals_filename, bvecs_filename)
    affine = dwis_img.affine
    dwis = dwis_img.get_data()

    print(" - Read mask image from disk...")
    mask_img = nib.load(mask_filename)
    mask = np.asarray(mask_img.get_data(), dtype=np.bool)
    dwis[np.logical_not(mask),...] = 0 
    
    output_dir = os.path.dirname(output_basename)
    if not output_dir:
        # If just a filename is provided, output_dir will be '' so make it cwd 
        output_dir = os.getcwd()
    if not os.path.isdir(output_dir):
        print(" - Make output directory {0}".format(output_dir))
        os.mkdir(output_dir)
    
    print(" - First, fit a standard tensor and calculate FA and MD")
    tensor_data = estimate_tensor(dwis, mask, bvals, bvecs)
    FA, MD, TR, AX, RAD = calculate_scalars(tensor_data, mask)

    if (wm_roi is not None) and (csf_roi is not None):
        print(" - Read ROIS corresponding to free water (CSF) and WM")
        csf_roi = np.asarray(nib.load(csf_roi).get_data(), dtype=bool)
        wm_roi = np.asarray(nib.load(wm_roi).get_data(), dtype=bool)    
    else:
        print(" - Need CSF and WM rois.")
        if exclude_mask:
            print(" - Exclude voxels in exclude mask .")
            exclude_mask = np.asarray(nib.load(exclude_mask).get_data(), dtype=bool)
        wm_roi, csf_roi = tissue_rois(mask, FA, TR, 
            erode_iterations=erode_iterations, fa_threshold=fa_threshold, 
            tr_threshold=tr_threshold, exclude=exclude_mask)
        
    init_f, init_tensor = initial_fit(dwis, bvals, bvecs, mask, wm_roi, csf_roi, MD, 
            csf_percentile=csf_percentile, wm_percentile=wm_percentile, 
            lmin=lmin, lmax=lmax, 
            evals_lmin=evals_lmin, evals_lmax=evals_lmax, md_value=md_value, 
            interpolate=interpolate, fixed_MD=fixed_MD)
        
    print(" - Save initial volume fraction image as Nifti")
    init_f_img = nifti_image(init_f, affine, cal_max=1)
    nib.save(init_f_img, '{0}_init_volume_fraction.nii.gz'.format(output_basename))

    print(" - Save initial tensor image as Nifti" )
    init_tensor_img = nifti_image(init_tensor, affine, intent_code=1005)
    nib.save(init_tensor_img, '{0}_init_tensor.nii.gz'.format(output_basename))

    print(" - Begin gradient descent.")
    final_f, final_tensor = gradient_descent(dwis, bvals, bvecs, mask, 
            init_f, init_tensor, niters=niters)
    
    print(" - Save tensor image as Nifti.")
    final_tensor_img = nifti_image(final_tensor, affine, intent_code=1005)
    nib.save(final_tensor_img, "{0}_fw_tensor.nii.gz".format(output_basename))

    print(" - Save volume fraction image as Nifti")
    final_f_img = nifti_image(final_f, affine, cal_max=1)
    nib.save(final_f_img, "{0}_fw_volume_fraction.nii.gz".format(output_basename))

    print(" - Save Fernet tensor scalars FA, TR, AX, RAD, FA difference as Nifti." )
    fw_fa, fw_md, fw_tr, fw_ax, fw_rad = calculate_scalars(final_tensor, mask)
    fw_fa_img = nifti_image(fw_fa, affine, cal_max=1)
    fw_tr_img = nifti_image(fw_tr, affine)
    fw_ax_img = nifti_image(fw_ax, affine)
    fw_rad_img = nifti_image(fw_rad, affine)
    fw_diff_fa_img = nifti_image(fw_fa - FA, affine)
    nib.save(fw_fa_img, "{0}_fw_tensor_FA.nii.gz".format(output_basename))
    nib.save(fw_tr_img, "{0}_fw_tensor_TR.nii.gz".format(output_basename))
    nib.save(fw_ax_img, "{0}_fw_tensor_AX.nii.gz".format(output_basename))
    nib.save(fw_rad_img, "{0}_fw_tensor_RAD.nii.gz".format(output_basename))
    nib.save(fw_diff_fa_img, "{0}_difference_FA.nii.gz".format(output_basename))
    
    print(" - Save WM and CSF rois as Nifti.")
    csf_roi_img = nifti_image(csf_roi.astype(np.int16), affine)
    wm_roi_img = nifti_image(wm_roi.astype(np.int16), affine)
    nib.save(csf_roi_img, "{0}_fw_csf_mask.nii.gz".format(output_basename))
    nib.save(wm_roi_img, "{0}_fw_wm_mask.nii.gz".format(output_basename))

    print(" - Done")
    
if __name__ == '__main__':
    main()
