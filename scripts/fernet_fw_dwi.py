#! /usr/bin/env python 
from __future__ import print_function 
import os
import numpy as np
import nibabel as nib
import argparse
from fernet.pipeline import * 
from fernet.utils import nifti_image, read_dwi, write_dwi 

DESCRIPTION = '''
    Apply free water elimination to a DWI image using an existing water volume fraction (VF) map. 
'''

d = 3.0e-3    
    
def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument('-d','-k','--data',action='store',metavar='dwi',dest='dwis_filename',
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
    p.add_argument('-f','--fraction',action='store',metavar='vf',dest='volume_fraction',
                    type=str, required=False, default=None, 
                    help='The free water volume fraction (VF) map.'
                    )
    p.add_argument('-o', '--output', action='store', metavar='output', dest='output',
                    type=str, required=True,
                    help='Output basename for corrected B0 map and corrected DWI.'
                    )
    return p

def main():    
    parser = buildArgsParser()
    args = parser.parse_args()
    dwis_filename = args.dwis_filename
    mask_filename = args.mask
    bvals_filename = args.bvals
    bvecs_filename = args.bvecs
    volume_fraction_filename = args.volume_fraction 
    output_basename = args.output
    fernet_correct_dwi(dwis_filename, bvals_filename, bvecs_filename, mask_filename, volume_fraction_filename, output_basename)
                
def fernet_correct_dwi(dwis_filename, bvals_filename, bvecs_filename, mask_filename, volume_fraction_filename, output_basename):
    
    print(" - Read DWIs from disk...")
    dwis_img, bvals, bvecs = read_dwi(dwis_filename, bvals_filename, bvecs_filename)
    affine = dwis_img.affine
    dwis = dwis_img.get_data()

    print(" - Read mask image from disk...")
    mask_img = nib.load(mask_filename)
    mask = np.asarray(mask_img.get_data(), dtype=np.bool)
    dwis[np.logical_not(mask),...] = 0 
    
    print(" - Read volume fraction image from disk...")
    vf_img = nib.load(volume_fraction_filename)
    fw_vf = vf_img.get_data()
    
    output_basename = os.path.realpath(output_basename)
    output_dir = os.path.dirname(output_basename)
    if not os.path.isdir(output_dir):
        print(" - Make output directory {0}".format(output_dir))
        os.mkdir(output_dir)
    
    b0 = np.mean(dwis[...,bvals==0], axis=-1)
    tissue_vf = 1 - fw_vf
    b0_corrected = b0 * tissue_vf 
    atten = dwis / b0[...,np.newaxis] 
    atten_tissue = (atten - (fw_vf[...,np.newaxis] * np.exp(-1*bvals*d)[np.newaxis,np.newaxis,np.newaxis,...])) / tissue_vf[...,np.newaxis]
    atten_tissue[np.isnan(atten_tissue)] = 0
    b0_corrected[np.isnan(b0_corrected)] = 0 
    dwi_corrected  = atten_tissue * b0_corrected[...,np.newaxis]
    dwi_corrected[np.isnan(dwi_corrected)] = 0 
    dwi_corrected = np.clip(dwi_corrected, 0, None)
    dwi_corrected[fw_vf > 0.9,...] = 0 # we don't trust the tissue compartment if water is greater than 90 % 
    dwi_corrected_img = nifti_image(dwi_corrected, affine)
    print(" - Write free water corrected DWI ")
    write_dwi(output_basename+'_fw_DWI.nii.gz', dwi_corrected_img, bvals, bvecs)
    print(" - Write free water corrected B0 ")
    b0_corrected_img = nifti_image(b0_corrected, affine)
    b0_corrected_img.to_filename(output_basename+'_fw_B0.nii.gz')
    
    print(" - Done")

if __name__ == '__main__':
    main()
