import os
import nibabel as nib
import numpy as np
from scipy.ndimage.morphology import binary_erosion

def erode_mask(mask_data, iterations=1):
    '''Erode a mask N times using scipy binary_erosion.'''
    mask_data = (mask_data > 0)*1
    mask_data[0,:,:] = 0 
    mask_data[:,0,:] = 0 
    mask_data[:,:,0] = 0 
    mask_data[-1,:,:] = 0
    mask_data[:,-1,:] = 0
    mask_data[:,:,-1] = 0
    mask_erode_data = (binary_erosion(mask_data, iterations=iterations))*1
    return mask_erode_data
    
def strip_nifti_ext(filename):
    '''Strips the extension off a nifti path and returns the rest.'''
    if filename.endswith('.nii.gz'):
        return filename[:-7]
    elif filename[-4:] in ['.nii','.hdr','.img']:
        return filename[:-4]
    else:
        raise IOError('Filename {} is not a Nifti path!'.format(filename))

def read_gradients(bval_file, bvec_file):
    '''Read bval and bvec files.
    
    Parameters 
    ----------
    bval_file : str
        Path to bval text file. If not provided, path will be inferred from the bvec filename.
    bvec_file : str
        Path to bval text file. If not provided, path will be inferred from the bval filename.
        
    Returns
    -------
    tuple
        A tuple of bvals, bvecs (numpy.ndarray)
    '''
    bvals = np.loadtxt(bval_file)
    bvecs = np.loadtxt(bvec_file).T
    return bvals, bvecs
        
def read_dwi(filename, bval_file=None, bvec_file=None):
    '''Read a nifti file and associated bval and bvec files.
    
    Parameters 
    ----------
    filename : str
        Path to diffusion weighted image NiFTI
    bval_file : Optional[str]
        Path to bval text file. If not provided, path will be inferred from the dwi filename.
    bvec_file : Optional[str]
        Path to bval text file. If not provided, path will be inferred from the dwi filename.
        
    Returns
    -------
    tuple
        A tuple of dwi_im (nibabel.Nifti1Image), bvals, bvecs (numpy.ndarray)
    '''
    dwi_im = nib.load(filename)
    if bval_file is None:
        bval_file = strip_nifti_ext(filename)+'.bval'
    if bvec_file is None:
        bvec_file = strip_nifti_ext(filename)+'.bvec'
    if not (os.path.exists(bval_file) and os.path.exists(bvec_file)):
        raise IOError('Cannot find corresponding bval/bvec files for DWI image')
    else:
        bvals, bvecs = read_gradients(bval_file, bvec_file) 
    return dwi_im, bvals, bvecs
    
def write_dwi(filename, dwi_img, bvals, bvecs):
    '''Write a nifti file and associated bval and bvec files.
    
    Parameters 
    ----------
    filename : str
        Path to output diffusion weighted image NiFTI
    dwi_img : nibabel.Nifti1Image
        DWI Nifti image object 
    bvals : numpy.ndarray
        bvals, a numpy ndarray of shape (N,) 
    bvecs : numpy.ndarray
        bvecs, a numpy ndarray of shape (N,3)
        
    Returns
    -------
    None
    '''
    dwi_img.to_filename(filename)
    np.savetxt(strip_nifti_ext(filename)+'.bval', bvals[None, :], fmt='%0.1f')
    np.savetxt(strip_nifti_ext(filename)+'.bvec', bvecs.transpose(), fmt='%0.6f')
    
def nifti_image(data, affine, hdr=None, intent_code=0, cal_max=None):
    ''' 
    Convenience function to create a Nifti image 
    Set intent_code to 1005 to save a tensor, 
    Set cal_max to 1 for an FA or volume fraction image.
    '''
    if hdr is None:
        hdr = nib.Nifti1Header()
    hdr.set_sform(affine)
    hdr.set_qform(affine)
    if cal_max is not None:
        hdr['cal_max'] = cal_max
    hdr['intent_code'] = intent_code
    return nib.Nifti1Image(data, affine, hdr)