# FERNET

FERNET is a diffusion MRI processing tool for running tissue signal estimation from diffusion-weighted imaging data. It can be run either through the provided Singularity/Apptainer container or directly with Python after installing the required dependencies.

The container-based method is recommended because it includes the required runtime environment.

## Requirements

### Container-based run

To run FERNET with the provided container, you need one of the following installed:

```bash
singularity
```

or

```bash
apptainer
```

The container image should be available as:

```bash
Fernet.sif
```

### Local Python run

To run FERNET without the container, install the following dependencies on your system.

Required software:

```text
Python >= 3.6
ANTs
```

FERNET requires ANTs to provide:

```bash
N4BiasFieldCorrection
```

Required Python packages:

```text
numpy
scipy
nibabel
pandas
scikit-learn
dipy
```

You can install the Python dependencies with:

```bash
python3 -m pip install numpy scipy nibabel pandas scikit-learn dipy
```

Make sure ANTs is available on your system path:

```bash
which N4BiasFieldCorrection
```

## Input files

FERNET requires a diffusion MRI image and a brain mask.

Required inputs:

```text
DWI image:   subject_DWI_preprocessed.nii.gz
Brain mask:  subject_tensor_mask.nii.gz
```

FERNET expects the corresponding `.bval` and `.bvec` files to be present using the same basename as the DWI image.

For example, if the DWI image is:

```text
subject_DWI_preprocessed.nii.gz
```

then the corresponding gradient files should be located in the same directory as:

```text
subject_DWI_preprocessed.bval
subject_DWI_preprocessed.bvec
```

## Command-line arguments

| Argument                    | Description                                            |
| --------------------------- | ------------------------------------------------------ |
| `-d`, `--data`              | Input DWI NIfTI image                                  |
| `-m`, `--mask`              | Brain mask NIfTI image                                 |
| `-o`, `--output`            | Output basename                                        |
| `-w`, `--wm`                | Optional white matter ROI                              |
| `-c`, `--csf`               | Optional CSF ROI                                       |
| `-x`, `--exclude`           | Optional exclusion mask                                |
| `-n`, `--niters`            | Number of gradient descent iterations; default is `50` |
| `-B`, `--bias-correct`      | Run N4 bias correction before FERNET                   |
| `-G`, `--no-gaussian-check` | Disable the Gaussian b-value range check               |

## Output basename

The `-o` argument should be an output basename, not only an output directory.

For example:

```bash
-o /path/to/output/sub-001
```

This tells FERNET to use:

```text
/path/to/output/sub-001
```

as the prefix for output files.

A good output value includes both the output directory and the subject/session prefix.

## Running with Singularity

General command:

```bash
singularity run --no-home \
  --bind /path/to/input_directory:/path/to/input_directory \
  --bind /path/to/output_directory:/path/to/output_directory \
  Fernet.sif \
  -d /path/to/input_directory/subject_DWI_preprocessed.nii.gz \
  -m /path/to/input_directory/subject_tensor_mask.nii.gz \
  -o /path/to/output_directory/subject
```

To enable N4 bias correction, add `-B`:

```bash
singularity run --no-home \
  --bind /path/to/input_directory:/path/to/input_directory \
  --bind /path/to/output_directory:/path/to/output_directory \
  Fernet.sif \
  -d /path/to/input_directory/subject_DWI_preprocessed.nii.gz \
  -m /path/to/input_directory/subject_tensor_mask.nii.gz \
  -o /path/to/output_directory/subject \
  -B
```

## Running with Apptainer

Apptainer uses nearly the same syntax as Singularity.

```bash
apptainer run --no-home \
  --bind /path/to/input_directory:/path/to/input_directory \
  --bind /path/to/output_directory:/path/to/output_directory \
  Fernet.sif \
  -d /path/to/input_directory/subject_DWI_preprocessed.nii.gz \
  -m /path/to/input_directory/subject_tensor_mask.nii.gz \
  -o /path/to/output_directory/subject
```

To enable N4 bias correction:

```bash
apptainer run --no-home \
  --bind /path/to/input_directory:/path/to/input_directory \
  --bind /path/to/output_directory:/path/to/output_directory \
  Fernet.sif \
  -d /path/to/input_directory/subject_DWI_preprocessed.nii.gz \
  -m /path/to/input_directory/subject_tensor_mask.nii.gz \
  -o /path/to/output_directory/subject \
  -B
```

## Hypothetical container example

Suppose the input files are stored in:

```text
/project/data/sub-001/DTI_Preprocess
```

with the following files:

```text
/project/data/sub-001/DTI_Preprocess/sub-001_DWI_preprocessed.nii.gz
/project/data/sub-001/DTI_Preprocess/sub-001_DWI_preprocessed.bval
/project/data/sub-001/DTI_Preprocess/sub-001_DWI_preprocessed.bvec
/project/data/sub-001/DTI_Preprocess/sub-001_tensor_mask.nii.gz
```

Suppose the desired output directory is:

```text
/project/results/fernet/sub-001
```

Then FERNET can be run with Singularity as:

```bash
singularity run --no-home \
  --bind /project/data/sub-001/DTI_Preprocess:/project/data/sub-001/DTI_Preprocess \
  --bind /project/results/fernet/sub-001:/project/results/fernet/sub-001 \
  Fernet.sif \
  -d /project/data/sub-001/DTI_Preprocess/sub-001_DWI_preprocessed.nii.gz \
  -m /project/data/sub-001/DTI_Preprocess/sub-001_tensor_mask.nii.gz \
  -o /project/results/fernet/sub-001/sub-001
```

The key point is that the output argument includes a subject-specific basename:

```bash
-o /project/results/fernet/sub-001/sub-001
```

## Example using CBICA-style paths

```bash
singularity run --no-home \
  --bind /cbica/home/username/input/sub-001/DTI_Preprocess:/cbica/home/username/input/sub-001/DTI_Preprocess \
  --bind /cbica/home/username/output/fernet/sub-001:/cbica/home/username/output/fernet/sub-001 \
  Fernet.sif \
  -d /cbica/home/username/input/sub-001/DTI_Preprocess/sub-001_DWI_preprocessed.nii.gz \
  -m /cbica/home/username/input/sub-001/DTI_Preprocess/sub-001_tensor_mask.nii.gz \
  -o /cbica/home/username/output/fernet/sub-001/sub-001
```

With N4 bias correction enabled:

```bash
singularity run --no-home \
  --bind /cbica/home/username/input/sub-001/DTI_Preprocess:/cbica/home/username/input/sub-001/DTI_Preprocess \
  --bind /cbica/home/username/output/fernet/sub-001:/cbica/home/username/output/fernet/sub-001 \
  Fernet.sif \
  -d /cbica/home/username/input/sub-001/DTI_Preprocess/sub-001_DWI_preprocessed.nii.gz \
  -m /cbica/home/username/input/sub-001/DTI_Preprocess/sub-001_tensor_mask.nii.gz \
  -o /cbica/home/username/output/fernet/sub-001/sub-001 \
  -B
```

## Running without a container

After installing the required software and Python dependencies, FERNET can be run directly with Python.

```bash
python3 scripts/fernet.py \
  -d /path/to/input_directory/subject_DWI_preprocessed.nii.gz \
  -m /path/to/input_directory/subject_tensor_mask.nii.gz \
  -o /path/to/output_directory/subject
```

With N4 bias correction enabled:

```bash
python3 scripts/fernet.py \
  -d /path/to/input_directory/subject_DWI_preprocessed.nii.gz \
  -m /path/to/input_directory/subject_tensor_mask.nii.gz \
  -o /path/to/output_directory/subject \
  -B
```

## Optional arguments

White matter ROI:

```bash
-w /path/to/wm_roi.nii.gz
```

CSF ROI:

```bash
-c /path/to/csf_roi.nii.gz
```

Exclusion mask:

```bash
-x /path/to/exclude_mask.nii.gz
```

Number of iterations:

```bash
-n 100
```

Disable Gaussian b-value range check:

```bash
-G
```

Example with optional ROIs:

```bash
singularity run --no-home \
  --bind /path/to/input_directory:/path/to/input_directory \
  --bind /path/to/output_directory:/path/to/output_directory \
  Fernet.sif \
  -d /path/to/input_directory/subject_DWI_preprocessed.nii.gz \
  -m /path/to/input_directory/subject_tensor_mask.nii.gz \
  -w /path/to/input_directory/subject_wm_roi.nii.gz \
  -c /path/to/input_directory/subject_csf_roi.nii.gz \
  -o /path/to/output_directory/subject \
  -n 100
```

