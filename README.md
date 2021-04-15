This Python package contains FERNET, a tool for Free-watER iNvariant 
Estimation of Tensor on single shell diffusion MR data. 

FERNET uses NIfTI as its data format for MR images, using the "nii.gz" 
file extension. It uses the FSL convention for "bval" and "bvec" text files. 

Installation requirements
-------------------------
- Python >= 2.7
- NumPy >= 1.10.4 
- Scipy >= 0.17.1
- Nibabel >= 2.0.1
- Dipy >= 0.10.1

Installation instruction
------------------------
* local install (user-wise, does not require root priviledge)
    - run "python setup.py install --prefix=~/local"
    - add "~/local/lib/python2.7/site-packages/" to your PYTHONPATH
    - add "~/local/bin" to your PATH

* system install (requires root privilege)
    - run "sudo python setup.py install"
