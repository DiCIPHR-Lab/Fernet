from distutils.core import setup
from distutils.extension import Extension
import numpy as np
import os

setup(name='fernet',
    description='Single shell free-water correction for Diffusion Tensor Imaging data.',
    version='0.1.dev0',
    author='William Parker',
    author_email='William.Parker@uphs.upenn.edu',
    url='http://www.med.upenn.edu/sbia/',
    scripts = [ os.path.join('scripts', 'fernet.py'),
                os.path.join('scripts', 'fernet_regions.py'),
                os.path.join('scripts', 'fernet_fw_dwi.py'),
                ],
    packages = ['fernet',
                ],
    )
