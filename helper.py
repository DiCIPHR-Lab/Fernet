# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 14:04:32 2016

@author: parkerwi
"""

import os,  sys, logging, argparse
import warnings
import numpy as np
import nibabel as nib
from scipy.ndimage.morphology import binary_erosion
from socket import gethostname
from getpass import getuser
import logging, tempfile, shutil
import subprocess, threading, signal
from datetime import datetime

def timestamp():
    return datetime.now().strftime('%Y%m%d-%H%M%S-%f')

def is_nifti_file(filename):
    """Returns True if filename is a nifti file. """
    try:
        nib.Nifti1Image.from_filename(filename)
    except:
        return False
    return True

def strip_nifti_ext(filename):
    '''Strips the extension off a nifti path and returns the rest.'''
    if filename.endswith('.nii.gz'):
        return filename[:-7]
    elif filename[-4:] in ['.nii','.hdr','.img']:
        return filename[:-4]
    else:
        raise ValueError(f'Filename {filename} is not a Nifti path!')

def is_valid_dwi(dwi_im, bvals, bvecs, raise_if_invalid=False):
    '''Check that a DWI image and associated bvals and bvecs are valid.
    
    DWI time dimension (N) must match bvals and bvecs 2nd dimensions. 
    bvals must be of shape (1,N)
    bvecs must be of shape (3,N)

    Parameters 
    ----------
    dwi_im : nibabel.Nifti1Image
        The DWI NiFTI image
    bvals : numpy.ndarray
        The bvals array
    bvecs : numpy.ndarray
        The bvecs array
    raise_if_invalid : Optional[bool]
        raise a ValueError if dwi is not valid.
        
    Returns
    -------
    bool
    '''
    bvecs_n = bvecs.shape[1]
    bvecs_dim = bvecs.shape[0]
    bvals_n = bvals.shape[1]
    bvals_dim = bvals.shape[0]
    dwi_n = 1 if len(dwi_im.shape)<4 else dwi_im.shape[-1]
    dimensions_valid = (dwi_n == bvals_n) and (dwi_n == bvecs_n) and (bvecs_dim == 3) and (bvals_dim == 1)
    if raise_if_invalid and not dimensions_valid:
        raise ValueError('Not a valid diffusion nifti and bval/bvec')
    return dimensions_valid

def check_inputs(*paths, nifti=False, directory=False, writable=False):
    '''For paths in args, check if inputs exist and are readable, and raise an Exception if not.
    
    Parameters
    ----------
    *paths : str
        Paths to check for existence and readability.
    nifti : bool
        If True, further check that the path is to a valid, readable Nifti file.
    directory : bool
        If True, further check that the path is a readable directory.
    writable : bool
        If True, further check that the path is writable. 
    
    Returns
    -------
    tuple
        Absolute paths of inputs
    '''
    logging.debug('diciphr.utils.check_inputs')
    if len(paths) == 0:
        raise ValueError('No files provided')
    
    not_exist=[]
    not_readable=[]
    not_writable=[]
    not_nifti=[]
    not_directory=[]
    for a in paths:
        logging.debug(f'Check input at path: {a}')
        if not os.access(a, os.F_OK):
            not_exist.append(a)
        elif not os.access(a, os.R_OK):
            not_readable.append(a)
        else:
            if writable and not os.access(a,os.W_OK):
                not_writable.append(a)
            if directory and not os.path.isdir(a):
                not_directory.append(a)
            if nifti and not is_nifti_file(a):
                not_nifti.append(a)
    if not_exist:
        raise FileNotFoundError('Paths do not exist: {}'.format(','.join(not_exist)))
    if not_readable:
        raise PermissionError('Paths are not readable: {}'.format(','.join(not_readable)))
    if not_writable:
        raise PermissionError('Paths are not writable: {}'.format(','.join(not_writable)))
    if not_directory:
        raise NotADirectoryError('Paths are not directories: {}'.format(','.join(not_directory)))
    if not_nifti:
        raise ValueError('Files are not valid nifti: {}'.format(','.join(not_nifti)))
    if len(paths) > 1:
        return tuple(os.path.realpath(_a) for _a in paths)
    else:
        return os.path.realpath(paths[0])
    
def make_dir(path, recursive=False, pass_if_exists=False):
    """
    Make a directory at the given path.
    
    Parameters:
    - path (str): The directory path to create.
    - recursive (bool): If True, create intermediate directories as needed.
    - pass_if_exists (bool): If True, do not raise an error if the directory already exists.
    - verbose (bool): If True, print status messages.
    
    Returns:
    - str: The path of the created (or existing) directory.
    """
    try:
        if recursive:
            os.makedirs(path, exist_ok=pass_if_exists)
        else:
            if pass_if_exists and os.path.exists(path):
                logging.debug(f"Directory already exists: {path}")
            else:
                os.mkdir(path)
        logging.info(f"Directory created: {path}")
    except FileExistsError:
        if not pass_if_exists:
            raise
        logging.debug(f"Directory already exists and was skipped: {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to create directory '{path}': {e}")
    
    return path

def get_slurm_jobid():
    return os.environ.get('SLURM_JOB_ID', 'interactive')

class MaxLevelFilter(logging.Filter):
    def __init__(self, max_level):
        self.max_level = max_level

    def filter(self, record):
        return record.levelno <= self.max_level
    

def set_up_logging(filename='', protocol_name='', debug=False, quiet=False, log_basics=False):
    """
    Set up logging to file, stdout (for INFO/DEBUG), and stderr (for WARNING+).
    """
    format_string = "%(asctime)s [{}] [%(levelname)-5.5s]  %(message)s".format(protocol_name or "%(threadName)-12.12s")
    formatter = logging.Formatter(format_string)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()  # Clear existing handlers
    root_logger.setLevel(logging.DEBUG if debug else (logging.ERROR if quiet else logging.INFO))

    # File handler
    if filename:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Stdout handler for DEBUG and INFO
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.addFilter(MaxLevelFilter(logging.INFO))
    stdout_handler.setFormatter(formatter)
    root_logger.addHandler(stdout_handler)

    # Stderr handler for WARNING and above
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)
    root_logger.addHandler(stderr_handler)
    
    if debug:
        logging.debug('Logging level set to DEBUG')
    elif quiet:
        logging.warning('Logging level set to ERROR')
    else:
        logging.info('Logging level set to INFO')

    if log_basics:
        logging.info(f'username: {getuser()}')
        logging.info(f'hostname: {gethostname()}')
        logging.info(f'cwd: {os.getcwd()}')
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        if slurm_job_id:
            logging.info(f'SLURM_JOB_ID: {slurm_job_id}')

def protocol_logging(protocol_name, directory=None, filename=None, debug=False, create_dir=True):
    """
    Set up the log file for a protocol named `protocol_name`.

    Parameters:
        protocol_name (str): Name of the protocol.
        directory (str, optional): Directory where the log file will be created.
        filename (str, optional): Full path to the log file. Overrides `directory` if provided.
        debug (bool, optional): If True, sets logging level to DEBUG and disables cleanup of temp directories.
        create_dir (bool, optional): If True, creates the directory if it doesn't exist.

    Returns:
        str: Path to the log file, or an empty string if no file is used.

    Raises:
        ValueError: If the directory or filename is not writable or invalid.
    """
    log_file = ''

    if filename:
        log_file = filename
        dir_path = os.path.dirname(log_file) or '.'
        if not os.path.exists(dir_path):
            if create_dir:
                os.makedirs(dir_path, exist_ok=True)
            else:
                raise ValueError(f"Directory for log file does not exist: {dir_path}")
        if not os.access(dir_path, os.W_OK):
            raise PermissionError(f"Directory is not writable: {dir_path}")
    elif directory:
        if not os.path.exists(directory):
            if create_dir:
                os.makedirs(directory, exist_ok=True)
            else:
                raise ValueError(f"Directory does not exist: {directory}")
        if not os.access(directory, os.W_OK):
            raise PermissionError(f"Directory is not writable: {directory}")
        _slurm_jid = get_slurm_jobid()
        _timestamp = timestamp()
        log_file = os.path.join(directory, f"{protocol_name}_{_slurm_jid}_{_timestamp}.log")
    set_up_logging(
        filename=log_file,
        protocol_name=protocol_name,
        debug=debug, 
        log_basics=True, 
    )
    return log_file

class DiciphrArgumentParser(argparse.ArgumentParser):
    '''
    A specialized ArgumentParser that adds three options common across all DiCIPHR scripts.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Flag to ensure options are added only once
        self._options_added = False  

    def _add_common_options(self):
        if self._options_added:
            return
        grp = self.add_argument_group('Other options')
        grp.add_argument(
            '--debug', action='store_true', dest='debug', 
            help='Debug mode: sets logging level, and temporary directories will not be deleted.'
        )
        grp.add_argument(
            '--workdir', action='store', metavar='<dir>', dest='workdir', type=str, default=None,
            help='The directory to preserve intermediate results. If not provided, a temporary directory will be created in $TMPDIR'
        )
        grp.add_argument(
            '--logfile', action='store', metavar='<file>', dest='logfile', type=str, default=None,
            help='A log file. If it does not exist, it will be created. Overrides --logdir'
        )
        grp.add_argument(
            '--logdir', action='store', metavar='<dir>', dest='logdir', type=str, default=None,
            help='A log directory. If it does not exist, it will be created, and a logfile will be created with name generated automatically'
        )
        self._options_added = True

    def parse_args(self, *args, **kwargs):
        self._add_common_options()
        args = super().parse_args(*args, **kwargs)
        # detect workdir 
        if args.workdir is not None:
            make_dir(args.workdir, recursive=True, pass_if_exists=True)
            set_diciphr_tmpdir(args.workdir)
        return args 

    def parse_known_args(self, *args, **kwargs):
        self._add_common_options()
        args, unknown = super().parse_known_args(*args, **kwargs)
        # detect workdir 
        if args.workdir is not None:
            make_dir(args.workdir, recursive=True, pass_if_exists=True)
            set_diciphr_tmpdir(args.workdir)
        return args, unknown
    
def set_diciphr_tmpdir(path):
    os.environ['DICIPHR_TMPDIR'] = os.path.realpath(path)

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
    bvals = bvals.reshape((1,np.size(bvals)))                       
    bvecs = np.loadtxt(bvec_file)
    # if a 3d volume bvecs will be shape (3,) and bvals will be shape (1,1)
    if len(bvecs.shape) == 1 and len(bvecs) == 3:
        bvecs = bvecs.reshape((3,1))      
    return bvals, bvecs


def read_dwi(filename, bval_file=None, bvec_file=None, force=False):
    '''Read a nifti file and associated bval and bvec files.
    
    Parameters 
    ----------
    filename : str
        Path to diffusion weighted image NiFTI
    bval_file : Optional[str]
        Path to bval text file. If not provided, path will be inferred from the dwi filename.
    bvec_file : Optional[str]
        Path to bval text file. If not provided, path will be inferred from the dwi filename.
    force : Optional[bool]
        If True, and bval/bvec files cannot be found, will return bvals and bvecs of all zeros. 
        Used for reading extra B0s for distortion correction with TOPUP which may lack accompanying bval/bvec files. 
    Returns
    -------
    tuple
        A tuple of dwi_im (nibabel.Nifti1Image), bvals, bvecs (numpy.ndarray)
    '''
    logging.debug('diciphr.diffusion.read_dwi')
    dwi_im = read_nifti(filename)
    if bval_file is None:
        bval_file = strip_nifti_ext(filename)+'.bval'
    if bvec_file is None:
        bvec_file = strip_nifti_ext(filename)+'.bvec'
    if not (os.path.exists(bval_file) and os.path.exists(bvec_file)):
        if force:
            if len(dwi_im.shape) ==3:
                N = 1 
            else:
                N = dwi_im.shape[-1]
            bvals = np.zeros((1,N))
            bvecs = np.zeros((3,N))
        else:
            raise FileNotFoundError('Cannot find bval/bvec files for DWI image')
    else:
        bvals, bvecs = read_gradients(bval_file, bvec_file) 
    is_valid_dwi(dwi_im, bvals, bvecs, True)
    return dwi_im, bvals, bvecs


def read_nifti(filename, dtype=None, lazy_load=True):
    '''Wrapper function for nibabel.load which also loads the data into memory if lazy_load is False
    Parameters 
    ----------
    filename : str
        Image filename
    nifti_im : Optional[nibabel.Nifti1Image]
        A nibabel Nifti1Image object.
    data : Optional[numpy.ndarray]
        A numpy array of image data. 
        To use, nifti_im should be None, and affine is required. 
    affine : Optional[numpy.ndarray]
        An affine transformation for nifti header. 
        
    Returns
    -------
    nibabel.Nifti1Image
    '''
    logging.debug('diciphr.utils.read_nifti')
    im = nib.load(filename)
    if dtype:
        data = im.get_fdata().astype(dtype)
        im = nib.Nifti1Image(data, im.affine, im.header)
        im.header.set_data_dtype(dtype)
    elif not lazy_load:
        data = im.get_fdata()
        im = nib.Nifti1Image(data, im.affine, im.header.copy())
    return im


def has_nifti_ext(filename):
    '''Returns true if the file ends with .hdr, .img, .nii, or .nii.gz .'''
    try:
        strip_nifti_ext(filename)
    except:
        return False
    return True
    

def write_nifti(filename, nifti_im=None, data=None, affine=None):
    '''Wrapper function for nibabel.save
    
    Parameters 
    ----------
    filename : str
        Output filename
    nifti_im : Optional[nibabel.Nifti1Image]
        A nibabel Nifti1Image object.
    data : Optional[numpy.ndarray]
        A numpy array of image data. 
        To use, nifti_im should be None, and affine is required. 
    affine : Optional[numpy.ndarray]
        An affine transformation for nifti header. 
        
    Returns
    -------
    str
        The output filename
    '''
    logging.debug('diciphr.utils.write_nifti')
    if not has_nifti_ext(filename):
        raise ValueError(f'Filename {filename} is not a valid Nifti path!')
    if nifti_im is None:
        nifti_im = nifti_image(data, affine)
    if affine is None:
        affine = nifti_im.header.get_best_affine()
    nifti_im.set_qform(affine, code=1)
    nifti_im.set_sform(affine, code=1)
    nifti_im.to_filename(filename)
    logging.info(f'Writing NiFTI to file {filename}')
    return filename

def write_dwi(filename,dwi_im,bvals,bvecs):
    '''Write a DWI NiFTI image and associated bvals and bvecs to file.
    
    Parameters 
    ----------
    filename : str
        The output nifti path with file extension.
    dwi_im : nibabel.Nifti1Image
        The DWI NiFTI image
    bvals : numpy.ndarray
        The bvals array
    bvecs : numpy.ndarray
        The bvecs array
        
    Returns
    -------
    None
    '''
    logging.debug('diciphr.diffusion.write_dwi')
    is_valid_dwi(dwi_im, bvals, bvecs, True)
    write_nifti(filename, dwi_im)
    bvalfile=strip_nifti_ext(filename)+'.bval'
    bvecfile=strip_nifti_ext(filename)+'.bvec'
    np.savetxt(bvalfile,bvals,fmt="%0.1f")
    np.savetxt(bvecfile,bvecs,fmt="%0.8f")

def nifti_image(data, affine, header=None, dtype=None, intent=0, **kwargs):
    '''Create a Nifti1Image instance. Casts data to a dtype if given.
    
    Parameters
    ----------
    data : numpy.ndarray
        The image data 
    affine : numpy.ndarray
        The affine transformation from voxel indices to real world coordinates.
    header : Optional[nibabel.Nifti1Header]
        The nifti header dictionary
    dtype : Optional[type]
        If provided, casts the data to this datatype and updates the header.
        
    Returns 
    -------
    nibabel.Nifti1Image
        The nifti image instance
    '''
    if dtype is not None:
        data = data.astype(dtype)
    else:
        dtype = data.dtype
        if dtype == np.dtype('int64'):
            logging.warning('Truncating int64 dtype to int32 because of Nifti incompatibility with 64-bit int')
            data = data.astype('int32')
        elif dtype == np.dtype('uint64'):
            logging.warning('Truncating uint64 dtype to uint32 because of Nifti incompatibility with 64-bit unsigned int')
            data = data.astype('uint32')
    im = nib.Nifti1Image(data, affine, header=header)
    im.header.set_sform(affine)
    im.header.set_qform(affine)
    im.header.set_data_dtype(data.dtype)
    im.header.set_intent(intent)
    for key in kwargs:
        im.header[key] = kwargs[key]
    im.update_header()
    return im

def which(program, raise_exception=True):
    """If an executable is at given path or found in PATH environmental variable, return path to command."""
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK) and (not os.path.isdir(fpath))

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ['PATH'].split(os.pathsep):
            path = path.strip('"').strip()
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    if raise_exception:
        raise EnvironmentError('Cannot find executable: {}'.format(program))
    return None


def make_temp_dir(directory=None, prefix='tmp'):
    '''Creates and returns a path to a unique working directory.

    If `directory` is None:
        - Use TMPDIR if set, else raise EnvironmentError

    Parameters
    ----------
    directory : Optional[str]
        Directory inside which to create a new tempdir
    prefix : Optional[str]
        Prefix of the new tempdir, default 'tmp'

    Returns
    -------
    str
        The newly created tempdir.
    '''
    if directory is None:
        directory = get_diciphr_tmpdir()
        if directory is None:
            raise EnvironmentError("No temporary directory specified. Environmental variable TMPDIR must be set.")    

    # Ensure the base directory exists
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception as e:
        logging.exception(f"Failed to create base temp directory {directory}: {e}")
        raise

    # Try to create a unique temp directory inside it
    jobid = os.environ.get('SLURM_JOB_ID','')
    if jobid:
        prefix = prefix+'_'+jobid 
    try:
        from uuid import uuid4
        workdir = tempfile.mkdtemp(prefix=prefix, dir=directory, suffix=str(uuid4()))
    except Exception as e:
        logging.warning(f"Failed to create temp dir with UUID suffix: {e}")
        try:
            workdir = tempfile.mkdtemp(prefix=prefix, dir=directory)
        except Exception as e2:
            logging.exception(f"Failed to create temp dir without UUID suffix: {e2}")
            raise

    logging.info(f"Created temporary directory {workdir}")
    return workdir

class TempDirManager:
    def __init__(self, directory=None, prefix='tmp', delete=True):
        self.tmpdir = make_temp_dir(directory, prefix)
        if get_diciphr_tmpdir():
            # User provided --workdir option - do not clean up 
            self.delete = False
        else:
            self.delete = delete 
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()
        
    def path(self):
        return self.tmpdir 
    
    def cleanup(self):
        if self.tmpdir and os.path.exists(self.tmpdir) and self.delete:
            try:
                logging.info(f"Deleting temp dir: {self.tmpdir}")
                shutil.rmtree(self.tmpdir)
            except Exception as e:
                logging.warning(f"Failed to delete temp dir {self.tmpdir}: {e}")
            finally:
                self.tmpdir = None 
        elif not self.delete:
            logging.info(f"Preserving temp dir: {self.tmpdir}")

class TerminationSignalReceived(Exception):
    """Custom exception to trigger cleanup on signal."""
    def __init__(self, signum):
        self.signum = signum 
        self.signal_name = signal.Signals(signum).name
        super().__init__(f"Received signal {self.signal_name} ({signum})")
        
    def __str__(self):
        return f"Signal {self.signal_name} ({self.signum}) received"

class ExecCommand:
    """
    A class to execute shell commands in SLURM or HPC environments,
    with support for environment variables, working directories, and real-time logging.

    Attributes:
        cmd_array (list): The command and arguments to execute.
        stdin (str): Optional input to pass to the command.
        environ (dict): Environment variables to set for the command.
        cwd (str): Working directory to run the command in.
        quiet (bool): If True, suppress logging output.
    """

    def __init__(self, cmd_array, stdin=None, environ=None, cwd=None, quiet=False):
        if not isinstance(cmd_array, (list, tuple)):
            raise TypeError("cmd_array must be a list or tuple of command arguments")
        self.cmd_array = list(map(str, cmd_array))
        self.stdin = stdin
        self.environ = environ or {}
        self.cwd = cwd
        self.quiet = quiet
        self.process = None 
        # Register SIGTERM and SIGINT handlers
        signal.signal(signal.SIGTERM, self._handle_termination_signal)
        signal.signal(signal.SIGINT, self._handle_termination_signal)
        signal.signal(signal.SIGXCPU, self._handle_termination_signal)
        signal.signal(signal.SIGUSR1, self._handle_termination_signal)
    
    def _handle_termination_signal(self, signum, frame):
        raise TerminationSignalReceived(signum)
    
    def _flush_logs(self):
        for handler in logging.root.handlers:
            try:
                handler.flush()
            except Exception as e:
                print(f"[WARN] Failed to flush log handler: {e}", file=sys.stderr)
                
    def run(self, raise_on_error=True):
        env = os.environ.copy()
        env.update(self.environ)
        logging.info("Run command:")
        logging.info(" ".join(self.cmd_array))
        if self.environ:
            logging.info(f"Provided environmental variables: {self.environ}")
        try:
            self.process = subprocess.Popen(
                self.cmd_array,
                stdin=subprocess.PIPE if self.stdin else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.cwd,
                env=env,
                universal_newlines=True,
                bufsize=1  # Line-buffered
            )

            def log_stream(stream, log_func):
                for line in iter(stream.readline, ''):
                    log_func(line.rstrip())
                stream.close()

            threads = []
            if not self.quiet:
                threads.append(threading.Thread(target=log_stream, args=(self.process.stdout, logging.info)))
                threads.append(threading.Thread(target=log_stream, args=(self.process.stderr, logging.warning)))

            for t in threads:
                t.start()

            if self.stdin:
                self.process.stdin.write(self.stdin)
                self.process.stdin.close()

            for t in threads:
                t.join()

            returncode = self.process.wait()

            if returncode != 0:
                if raise_on_error:
                    raise subprocess.CalledProcessError(returncode, self.cmd_array)
                else:
                    logging.warning(f"Command failed with return code {returncode}")

            self._flush_logs()
            return returncode

        except subprocess.CalledProcessError as e:
            logging.exception(f"Command failed with return code {e.returncode}")
            return e.returncode
        
        except TerminationSignalReceived as e:
            logging.warning(f"Terminating due to signal: {e.signal_name}")
            if self.process:
                try:
                    self.process.terminate()
                    self.process.wait(timeout=5)
                    logging.info(f"Subprocess terminated cleanly after signal {e.signal_name}.")
                except Exception as ex:
                    logging.error(f"Failed to terminate subprocess: {ex}")
            self._flush_logs()
            raise 
        
        except Exception as e:
            logging.exception("Command execution failed")
            raise RuntimeError("Command execution failed") from e

def get_diciphr_tmpdir():
    return os.environ.get('DICIPHR_TMPDIR', os.environ.get('TMPDIR'))

def extract_b0(dwi_im, bvals, bvecs=None, first=False, average=True):
    '''Extract the average B0 from a dwi volume. 
    
    bvals are used, and bvecs are not, but the argument is provided, 
    to preserve usage relative to other dwi functions in this module. 
    
    Parameters
    ----------
    dwi_im : nibabel.Nifti1Image
        The DWI NiFTI image
    bvals : numpy.ndarary
        The bvals array
    bvecs : Optional[numpy.ndarray]
        The bvecs array. Not used
    first : Optional[bool]
        If True, return the first encountered B0. 
    average: Optional[bool]
        If True ( default ) averages the B0s, else returns a 4D volume of B0 images.
    Returns
    -------
    nibabel.Nifti1Image
        The average B0 image.
    '''
    logging.debug('diciphr.diffusion.extract_b0')
    data = dwi_im.get_fdata()
    if len(data.shape)==3:
        data = data[...,np.newaxis]
    bvals = np.loadtxt(bvals)
    bvals = np.asarray(bvals, dtype=float).squeeze()
    indy = np.where(np.isclose(bvals, 0.0))[0]
    if len(indy) == 0:
        raise ValueError(f"No b0 volumes found. bvals={bvals}")
    if first:
        b0 = data[..., indy[0]]
    else:
        b0 = data[..., indy]
        if average:
            b0 = np.mean(b0, axis=-1)
    b0_im = nifti_image(b0, dwi_im.affine, dwi_im.header)
    return b0_im


def n4_bias_correct_dwi(dwi_img, bvals, bvecs, field=False, mask_img=None, weight_img=None,
                convergence=None, bspline=None):
    ''' 
    Run ANTs N4BiasFieldCorrection on a DWI image.
    '''
    logging.debug('diciphr.diffusion.n4_bias_correct_dwi')
    if convergence is None:
        convergence = '[50x50x50x50,1e-3]'
    if bspline is None:
        bspline = '[150,3]'
    with TempDirManager(prefix='n4biasdwi') as manager:
        tmpdir = manager.path()
        b0_filename = os.path.join(tmpdir, 'b0.nii')
        outimg_fn = os.path.join(tmpdir, 'output.nii')
        outfield_fn = os.path.join(tmpdir, 'field.nii')
        # extract b0 
        dwi_img = nib.load(dwi_img)
        b0_im = extract_b0(dwi_img, bvals)
        b0_data = b0_im.get_fdata()
        percLow = np.percentile(b0_data[b0_data>0],0.5)
        percHigh = np.percentile(b0_data[b0_data>0],99.5)
        b0_winsor = b0_data.copy()
        b0_winsor[b0_winsor < percLow] = percLow
        b0_winsor[b0_winsor > percHigh] = percHigh
        write_nifti(b0_filename, nifti_image(b0_winsor, b0_im.affine))
        cmd = [ 'N4BiasFieldCorrection', '-d', '3', '-i', b0_filename,
                '-r', '-b', str(bspline),'-c', str(convergence), '-v', '1',
                '-o', f'[{outimg_fn},{outfield_fn}]'
        ]
        if mask_img is not None:
            mask_filename = os.path.join(tmpdir, 'mask.nii')
            write_nifti(mask_filename, mask_img)
            cmd += ['-x', mask_filename]
        if weight_img is not None:
            weight_filename = os.path.join(tmpdir, 'weight.nii')
            write_nifti(weight_filename, weight_img)
            cmd += ['-w',weight_filename]
        ExecCommand(cmd).run()
        field_im = read_nifti(outfield_fn, lazy_load=False)
        # Correct each volume of the dwi 
        dwi_data = dwi_img.get_fdata()
        field_data = field_im.get_fdata()
        logging.info('Divide DWI image every volume by bias field map')
        dwi_corr = dwi_data / field_data[...,None]
        dwi_corr_im = nifti_image(dwi_corr, dwi_img.affine)
    if field:
        return (dwi_corr_im, bvals, bvecs), field_im 
    else:
        return dwi_corr_im, bvals, bvecs

def remove_dwi_gradients(dwi_im, bvals, bvecs, list_of_indices):
    '''Remove gradient images and associated bvectors from a DWI image.
    
    Given a dwi image, bvals, bvecs and a list of indices, 
    will remove subimages at those indices from the dwi image volume 
    as well as the associated elements from the bval and bvec arrays.
    
    Parameters
    ----------
    dwi_im : nibabel.Nifti1Image
        The DWI NiFTI image
    bvals : numpy.ndarray
        The bvals array
    bvecs : numpy.ndarray
        The bvecs array
    list_of_indices : list
        The indices to remove.
        
    Returns
    -------
    tuple
        A tuple of dwi_im (nibabel.Nifti1Image), bvals, bvecs (numpy.ndarray)
    '''
    is_valid_dwi(dwi_im, bvals, bvecs, True) # raises if not
    data=dwi_im.get_fdata()
    if len(list_of_indices) < 1:
        logging.warning('Asked to remove an empty list of gradient images. Returning original DWI')
        return (dwi_im, bvals, bvecs )
    max_index = max(list_of_indices)
    if max_index >= data.shape[-1]:
        raise ValueError(f'Cannot remove dwi gradient # {max_index} from {data.shape[-1]} gradients')
    
    bvals_out=bvals.reshape((1,np.size(bvals)))
    bvecs_out=bvecs.copy()
    # are next 2 lines needed? probably not 
    if not (np.size(bvals_out) == np.size(bvecs_out)/3 and np.size(bvals_out) == data.shape[-1]): 
        raise ValueError('Bval, bvec and data do not encode the same number of gradients!')
    list_of_indices=sorted(list_of_indices,reverse=True)
    for ind in list_of_indices:
        data = np.delete(data,ind,axis=3)
        bvals_out=np.delete(bvals_out,ind,axis=1)
        bvecs_out=np.delete(bvecs_out,ind,axis=1)
    hdr=dwi_im.header
    hdr['dim'][4] -= len(list_of_indices)
    dwi_image_out=nifti_image(data,dwi_im.affine,header=hdr)
    return (dwi_image_out,bvals_out,bvecs_out)    

def extract_shells_from_multishell_dwi(dwi_im, bvals, bvecs, target_bvalues, error=20):
    ''' Extract the gradients from a diffusion weighted image that are within error of target bvalues.
    
    Parameters
    ----------
    dwi_im : nibabel.Nifti1Image
        The DWI image
    bvals : numpy.ndarray
        The 1xN bvals array
    bvecs : numpy.ndarray
        The 3xN bvecs array
    target_bvalues : list 
        A list of bvalues to keep.
    error : Optional[int]
        Keep gradients with bvalues within this much of the target. 
        
    Returns
    -------
    tuple
        A tuple of dwi_im (nibabel.Nifti1Image), bvals, bvecs (numpy.ndarray)
    '''
    logging.debug('diciphr.diffusion.extract_shells_from_multishell_dwi')
    if not hasattr(target_bvalues, '__getitem__'):
        target_bvalues=[target_bvalues]
    indices_to_remove = set(range(np.size(bvals)))
    for bvalue in target_bvalues:
        indices = set(np.where(np.abs(bvals - bvalue) > error)[1])
        indices_to_remove.intersection_update(indices)
    indices_to_remove = sorted(list(indices_to_remove))
    if len(indices_to_remove) == 0:
        logging.info('No indices to remove. Returning input DWI data.')
        return dwi_im, bvals, bvecs
    return remove_dwi_gradients(dwi_im, bvals, bvecs, indices_to_remove)

def has_gaussian_shells(bvals):
    return (np.logical_and(np.array(bvals)<=1500, np.array(bvals)>=500)).any()

def extract_gaussian_shells(dwi_im, bvals, bvecs):
    logging.debug('diciphr.diffusion.extract_gaussian_shells')
    if np.max(bvals) > 1500 or np.min(bvals[bvals>0]) < 500:
        logging.info("B-values outside the Gaussian range 500-1500 detected.")
        bvals_gaussian = list(np.unique(bvals[np.logical_and(bvals >= 500, bvals <=1500)]))
        if len(bvals_gaussian) == 0 :
            logging.info("No b-values between 500 and 1500 detected. Returning original DWI")
            return dwi_im, bvals, bvecs 
        else:
            shells = [0] + bvals_gaussian
            logging.info("Extracted b-values: {}".format(shells))
            return extract_shells_from_multishell_dwi(dwi_im, bvals, bvecs, shells)
    else:
        logging.info("B-values already inside the Gaussian range 500-1500.")
        return dwi_im, bvals, bvecs 
    

def round_bvals(bvals,nearest='hundred'):
    '''Round a bvals array to the specified significance.
    
    Parameters
    ----------
    bvals : numpy.ndarary
        The bvals array
    nearest : Optional[str]
        One of 'thousand','hundred','ten','one'
    
    '''
    nearest_dict = {'one':1.,
                    'ten':10.,
                    'hundred':100.,
                    'thousand':1000.}
    factor = nearest_dict[nearest]
    bvals = bvals / factor
    bvals = np.round(bvals)
    bvals = bvals * factor
    return bvals
