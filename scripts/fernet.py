#! /usr/bin/env python
import os, sys, logging
import shutil
from pathlib import Path
import nibabel as nib
FERNET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FERNET_ROOT))
from fernet.pipeline import run_fernet
from helper import check_inputs, make_dir, protocol_logging, DiciphrArgumentParser, n4_bias_correct_dwi


DESCRIPTION = '''
FERNET : FreewatER EstimatoR using iNtErpolated iniTialization
Initialize the volume fraction map and free-water corrected tensor.
'''

PROTOCOL_NAME = 'FERNET' 

# ADVANCED FERNET PARAMETERS
fernet_kwargs = {
    'erode_iterations' : 8,
    'fa_threshold' : 0.7,
    'tr_threshold' : 0.0085,
    'md_value' : 0.6e-3,
    'lmin' : 0.1e-3,
    'lmax' : 2.5e-3,
    'evals_lmin' : 0.1e-3,
    'evals_lmax' : 2.5e-3,
    'wm_percentile' : 5,
    'csf_percentile' : 95,
    'interpolate' : True,
    'fixed_MD' : False,
    'weight' : 100.0,
    'step_size' : 1.0e-7
}

def buildArgsParser():
    p = DiciphrArgumentParser(description=DESCRIPTION)
    p.add_argument('-d', '--data', action='store', metavar='dwi', dest='dwi',
                    type=str, required=True, 
                    help='Input DWIs data file (Nifti or Analyze format).'
                    )
    p.add_argument('-o', '--output', action='store', metavar='output', dest='output',
                    type=str, required=True,
                    help='Output basename for init tensor map and volume fraction.'
                    )
    p.add_argument('-m', '--mask', action='store', metavar='mask', dest='mask',
                    type=str, required=True, 
                    help='Brain mask file (Nifti or Analyze format).'
                    )
    p.add_argument('-r', '--bvecs', action='store', metavar='bvecs', dest='bvecs',
                    type=str, required=False, default=None,
                    help='Gradient directions (.bvec file).'
                    )
    p.add_argument('-b', '--bvals', action='store', metavar='bvals', dest='bvals',
                    type=str, required=False, default=None,
                    help='B-values (.bval file).'
                    )
    p.add_argument('-w', '--wm', action='store', metavar='roi',dest='wm_roi',
                    type=str, required=False, default=None, 
                    help='An ROI defined in deep WM, e.g. corpus callosum (Nifti or Analyze format).'
                    )
    p.add_argument('-c', '--csf', action='store', metavar='roi',dest='csf_roi',
                    type=str, required=False, default=None, 
                    help='An ROI defined in CSF, e.g. ventricle (Nifti or Analyze format).'
                    )
    p.add_argument('-x', '--exclude', action='store', metavar='mask',dest='exclude_mask',
                    type=str, required=False, default=None, 
                    help='A mask (e.g. peritumoral region) of voxels to exclude when getting typical WM, GM voxels.'
                    )
    p.add_argument('-n', '--niters', action='store', metavar='niters', dest='niters', 
                    type=int, required=False, default=50,
                    help='Number of iterations of the gradient descent. Default is 50'
                    )
    p.add_argument('-B', '--bias-correct', action='store_true', dest='bias_correct', 
                    help='Run bias correction before FERNET. Recommended if data has not already been bias-corrected (as it typically is by dti_preprocess.py)'
                    )
    p.add_argument('-G', '--no-gaussian-check', action='store_false', dest='gaussian', 
                    help='FERNET should be run on data with b-values between 500-1500 s/mm2. If your data does not fall in this range, this option overrides the check (not recommended)'
                    )
    return p

def main(argv):    
    parser = buildArgsParser()
    args = parser.parse_args(argv)
    protocol_logging(PROTOCOL_NAME, directory=args.logdir, filename=args.logfile, debug=args.debug, create_dir=True)
    try:
        output_dir = os.path.dirname(os.path.realpath(args.output))
        make_dir(output_dir,recursive=True,pass_if_exists=True)
        check_inputs(args.dwi, nifti=True)
        check_inputs(args.mask, nifti=True)
        check_inputs(args.mask, nifti=True)
        os.makedirs(args.output, exist_ok=True)
        if args.exclude_mask:
            check_inputs(args.exclude_mask, nifti=True)
        if args.wm_roi:
            check_inputs(args.wm_roi, nifti=True)
        if args.csf_roi:
            check_inputs(args.csf_roi, nifti=True)
        if args.bias_correct:
            base = args.dwi
            if base.endswith(".nii.gz"):
                base = base[:-7]
            elif base.endswith(".nii"):
                base = base[:-4]
            bval_input = base + ".bval"
            bvec_input = base + ".bvec"
            print(bval_input)
            print(bvec_input)
            n4_dwi, n4_bval, n4_bvec = n4_bias_correct_dwi(args.dwi, bval_input, bvec_input)
            n4_dwi_out = os.path.join(args.output, "n4_dwi.nii.gz")
            n4_bval_out = os.path.join(args.output, "n4_dwi.bval")
            n4_bvec_out = os.path.join(args.output, "n4_dwi.bvec")
            nib.save(n4_dwi, n4_dwi_out)
            shutil.copyfile(n4_bval, n4_bval_out)
            shutil.copyfile(n4_bvec, n4_bvec_out)
            run_fernet(n4_dwi_out, n4_bval_out, n4_bvec_out, args.mask, args.output,wm_roi=args.wm_roi, csf_roi=args.csf_roi, exclude_mask=args.exclude_mask, niters=args.niters, gaussian=args.gaussian, **fernet_kwargs)
        if not args.bias_correct:
            run_fernet(args.dwi, args.bvals, args.bvecs, args.mask, args.output,
                    wm_roi=args.wm_roi, csf_roi=args.csf_roi, exclude_mask=args.exclude_mask, 
                    niters=args.niters, gaussian=args.gaussian, bias_correct=args.bias_correct, **fernet_kwargs)
        print(args.output)
    except Exception:
        logging.exception(f"Exception encountered running {PROTOCOL_NAME}")
        raise
    
if __name__ == '__main__':
    main(sys.argv[1:])
