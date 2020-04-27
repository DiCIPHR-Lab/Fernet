#!/usr/bin/env python

import os, sys, argparse
import nibabel as nib 
import numpy as np
from sklearn.metrics import mean_squared_error
from dipy.sims.voxel import multi_tensor
from dipy.core.gradients import gradient_table
from dipy.core.sphere import disperse_charges, HemiSphere

####################################################
_physical_parameters = {
    'wm' : {
        't1' : {'mean' : 0.832,   'stddev' : 0.010},
        't2' : {'mean' : 79.6e-3, 'stddev' : 0.6e-3,},
        'rho' : 0.65,
    },
    'gm' : {
        't1' : {'mean' : 1.331,   'stddev' : 0.013,},
        't2' : {'mean' : 110.e-3, 'stddev' : 2.0e-3,},
        'rho' : 0.75,
    },
    'csf' : {
        't1' : {'mean' : 3.5,     'stddev' : 0.1,},
        't2' : {'mean' : 0.25,    'stddev' : 0.01,},
        'rho' : 1.0,
    },
}

# The value of free water diffusion is set to its known value
Dwater = 3e-3
    
###################################################

def mr_signal(water_vf, te, tr):
    """
    Computes MR image, provided images of the WM, GM, CSF and background volume
    fractions.
    Parameters
    ----------
    wm_vf : array-like, shape ``(dim_x, dim_y, dim_z)``
        White matter volume fraction.
    wm_t1 : array-like, shape ``(dim_x, dim_y, dim_z)``
        White matter t1 relaxation image.
    wm_t2 : array-like, shape ``(dim_x, dim_y, dim_z)``
        White matter t2 relaxation image.
    gm_vf : array-like, shape ``(dim_x, dim_y, dim_z)``
        Gray matter volume fraction
    gm_t1 : array-like, shape ``(dim_x, dim_y, dim_z)``
        Gray matter t1 relaxation image.
    gm_t2 : array-like, shape ``(dim_x, dim_y, dim_z)``
        Gray matter t2 relaxation image.
    csf_vf : array-like, shape ``(dim_x, dim_y, dim_z)``
        CSF volume fraction
    csf_t1 : array-like, shape ``(dim_x, dim_y, dim_z)``
        CSF t1 relaxation image.
    csf_t2 : array-like, shape ``(dim_x, dim_y, dim_z)``
        CSF t2 relaxation image.
    background_vf : array-like, shape ``(dim_x, dim_y, dim_z)``
        Background volume fraction
    te : double
        echo time (s)
    tr : double
        repetition time (s)
    Returns
    -------
    image : array-like, shape ``(dim_x, dim_y, dim_z)``
        The computed MR signal.
    """
    wm_rho = _physical_parameters['wm']['rho']
    csf_rho = _physical_parameters['csf']['rho']
    wm_t1 = _physical_parameters['wm']['t1']['mean']
    wm_t2 = _physical_parameters['wm']['t2']['mean']
    csf_t1 = _physical_parameters['csf']['t1']['mean']
    csf_t2 = _physical_parameters['csf']['t2']['mean']
    image = ((1-water_vf) * wm_rho * (1.0 - np.exp(-1.0 * tr / wm_t1)) * np.exp(-1.0 * te / wm_t2)) + (water_vf * csf_rho * (1.0 - np.exp(-tr / csf_t1)) * np.exp(-te / csf_t2))
    return image

def rician_noise(image, sigma, rng=None):
    """
    Add Rician distributed noise to the input image.
    Parameters
    ----------
    image : array-like, shape ``(dim_x, dim_y, dim_z)`` or ``(dim_x, dim_y,
        dim_z, K)``
    sigma : double
    rng : random number generator (a numpy.random.RandomState instance).
    """
    if rng is None:
        rng = np.random.RandomState(None)
    n1 = rng.normal(loc=0, scale=sigma, size=image.shape)
    n2 = rng.normal(loc=0, scale=sigma, size=image.shape)
    return np.sqrt((image + n1)**2 + n2**2)

def get_directions(nDTdirs):
    theta = np.pi * np.random.rand(nDTdirs)
    phi = 2 * np.pi * np.random.rand(nDTdirs)
    hsph_initial = HemiSphere(theta=theta, phi=phi)
    hsph_updated, potential = disperse_charges(hsph_initial, 5000)
    return hsph_updated.vertices
    
def generate_dwi(bvals, bvecs, b0, csf, l1, l2, l3, snr, tr, te, VF, DTdirs, nrep):
    # ----------------------------------------------------------------
    print('Generating simulations...')
    # ----------------------------------------------------------------
    gtab = gradient_table(bvals, bvecs) 
    DWI = np.empty((VF.size, len(DTdirs), nrep, bvals.size))
    mevals = np.array([[l1, l2, l3],
                       [Dwater, Dwater, Dwater]])
    for i, vf in enumerate(VF):
        # estimating volume fractions for both simulations
        # compartments
        fractions = [100 - vf, vf]
        water_vf = vf/100.0
        S0_b = (csf * water_vf) + ((1 - water_vf) * b0)
        for j, vec in enumerate(DTdirs):
            # Repeat simulations for the given directions
            for k in range(nrep):
                # print(i,j,k)
                # Multi-compartmental simulations are done using
                # Dipy's function multi_tensor
                signal, sticks = multi_tensor(gtab, mevals,
                                              S0=S0_b,
                                              angles=[vec, (1, 0, 0)],
                                              fractions=fractions,
                                              snr=snr)
                # if i==0:
                    # t1 = csf_t1
                    # t2 = csf_t2
                    # rho =  csf_rho
                # elif i!=0:
                    # t1 = wm_t1
                    # t2 = wm_t2
                    # rho = wm_rho
                # s0 = rho * (1.0 - np.exp(-tr / t1)) * np.exp(-te / t2) 
                # sigma = s0*S0_b/snr
                # DWI[i, j, k, :] = rician_noise(signal, sigma, rng=args.rng)				
                DWI[i, j, k, :] = signal			
        
    #save the dwi image			
    _affine = np.eye(4)
    _hdr = nib.Nifti1Header()
    _hdr.set_sform(_affine)
    _hdr.set_qform(_affine)
    
    return nib.Nifti1Image(DWI, _affine, _hdr) 

def parse_args():
    #collect input params
    description="This script simulates DWIs from dipy multi-tensor models coupled with b0 reflective of tissues properties."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-bvals', dest='bvals', required=True,
                        help="bvals file for simulation.")
    parser.add_argument('-bvecs', dest='bvecs', required=True,
                        help="bvecs file for simulation.")								
    parser.add_argument('-snr', dest='snr', required=True, type=float, 
                        help="SNR desired for simulation.")									
    parser.add_argument('-b0', dest='b0', required=True, type=float, 
                        help="Simulated voxel B0 value.")	
    parser.add_argument('-eig', dest='eig', required=True, type=float, nargs=3,
                        help="The eigenvalues of simulated voxel.")		                    
    parser.add_argument('-csf', dest='csf', required=False, default=2152.0, type=float, 
                        help="Simulated reference CSF B0 value.")			
    parser.add_argument('-te', dest='te', required=False, default=0.115, type=float, 
                        help="Echo Time for simulation.")	
    parser.add_argument('-tr', dest='tr', required=False, default=10.0, type=float, 
                        help="Repetition Time for simulation.")	
    parser.add_argument('-nrep', dest='nrep', required=False, default=100, type=int, 
                        help="Number of repetitions in simulation.")	
    parser.add_argument('-rng', dest='rng', required=False, default=None, 
                        help="Random number generator seed")		
    parser.add_argument('-outdir', dest='outdir', required=False, default='SimulatedData', 
                        help="Output Directory")    
    return parser.parse_args()
    
def main():
    args = parse_args()
    VF = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    nrep = 100
    nDTdirs = 100
    b0 = args.b0 
    csf = args.csf 
    l1, l2, l3 = args.eig 
    snr = args.snr 
    te = args.te 
    tr = args.tr 
    bvals = np.loadtxt(args.bvals)
    bvecs = np.loadtxt(args.bvecs)
    outdir = args.outdir
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    output_prefix='{0}/simulated-B0_{1}-L1_{2}-L2_{3}-L3_{4}-SNR_{5:g}-TE_{6}-TR_{7}'.format(
        outdir, b0, l1, l2, l3, snr, te, tr
    )
    # These directions are sampled using the same procedure used
    # to evenly sample the diffusion gradient directions
    DTdirs = get_directions(nDTdirs)
    #save groud truth directions
    DTdirsfile = '{}.GT_Directions'.format(output_prefix)
    np.savetxt(DTdirsfile, DTdirs, fmt="%0.8f")

    dwi_img = generate_dwi(bvals, bvecs, b0, csf, l1, l2, l3, snr, tr, te, VF, DTdirs, nrep)
    dwi_file = '{0}.nii.gz'.format(output_prefix)
    nib.save(dwi_img, dwi_file)
    print ('DONE!')

if __name__ == '__main__':
    main()