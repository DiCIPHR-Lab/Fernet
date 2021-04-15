#! /usr/bin/env python 
from __future__ import print_function
import numpy as np
import nibabel as nib
import argparse
from fernet.pipeline import * 
from fernet.utils import nifti_image, read_dwi

DESCRIPTION = '''
Creates WM and CSF rois for running FERNET. 
'''

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
    p.add_argument('-x','--exclude',action='store',metavar='mask',dest='exclude_mask',
                    type=str, required=False, default=None, 
                    help='A mask (e.g. peritumoral region) of voxels to exclude when getting typical WM, GM voxels.'
                    )
    p.add_argument('-o', '--output', action='store', metavar='output', dest='output',
                    type=str, required=True,
                    help='Output basename for rois.'
                    )
    p.add_argument('-f','--fa-threshold',action='store',metavar='fa',dest='fa_threshold',
                    type=float, required=False, default=0.7, 
                    help='The FA threshold to define the WM roi. Default is 0.7'
                    )
    p.add_argument('-t','--tr-threshold',action='store',metavar='tr',dest='tr_threshold',
                    type=float, required=False, default=0.0085, 
                    help='The TR threshold to define the CSF roi. Default is 0.0085'
                    )
    p.add_argument('-n','--erode-iters',action='store',metavar='N',dest='erode_iterations',
                    type=int, required=False, default=8, 
                    help='Erode the mask this many iterations to narrow in on ventricles and deep WM. Default is 8'
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
    output_basename = args.output
    fa_threshold = args.fa_threshold
    tr_threshold = args.tr_threshold
    erode_iterations = args.erode_iterations
    fernet_regions(dwis_filename, bvals_filename, bvecs_filename, mask_filename, output_basename,
                exclude_mask=exclude_mask, fa_threshold=fa_threshold, tr_threshold=tr_threshold, erode_iterations=erode_iterations)
                
def fernet_regions(dwis_filename, bvals_filename, bvecs_filename, mask_filename, output_basename,
                exclude_mask=None, fa_threshold=0.7, tr_threshold=0.0085, erode_iterations=8):
    print(" - Read DWIs from disk...")
    dwis_img, bvals, bvecs = read_dwi(dwis_filename, bvals_filename, bvecs_filename)
    affine = dwis_img.affine
    dwis = dwis_img.get_data()
    print(" - Read mask image from disk...")
    mask_img = nib.load(mask_filename)
    mask = np.asarray(mask_img.get_data(), dtype=np.bool)
    dwis[np.logical_not(mask),...] = 0 
    print(" - First, fit a standard tensor and calculate FA and MD" )
    tensor_data = estimate_tensor(dwis, mask, bvals, bvecs)
    FA, MD, TR, AX, RAD = calculate_scalars(tensor_data, mask)
    if exclude_mask:
        print(" - Exclude voxels in exclude mask .")
        exclude_mask = np.asarray(nib.load(exclude_mask).get_data(), dtype=bool)
    print(" - Erode mask {0} times".format(erode_iterations))
    print(" - Threshold FA at {0} and TR at {1}".format(fa_threshold, tr_threshold))
    wm_roi, csf_roi = tissue_rois(mask, FA, TR, 
        erode_iterations=erode_iterations, fa_threshold=fa_threshold, 
        tr_threshold=tr_threshold, exclude=exclude_mask)
    wm_roi_img = nifti_image(wm_roi.astype(np.int16), affine)
    csf_roi_img = nifti_image(csf_roi.astype(np.int16), affine)
    print(" - Save ROIs ")
    nib.save(wm_roi_img, "{0}_wm_mask.nii.gz".format(output_basename))
    nib.save(csf_roi_img, "{0}_csf_mask.nii.gz".format(output_basename))
    print(" - Done")

if __name__ == '__main__':
    main()
