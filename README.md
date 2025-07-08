# Neural Inverse Operators (NIO) for Solving PDE Inverse Problems

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the paper [**Neural Inverse Operators for Solving PDE Inverse Problems**](https://openreview.net/pdf?id=S4fEjmWg4X).

<img src="NIORB.png" width="800" alt="NIO Architecture">

## Table of Contents
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Available Models](#available-models)
- [Datasets](#datasets)
- [Training](#training)
- [Citation](#citation)
- [License](#license)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/nio-jma.git
   cd nio-jma
   ```

2. **Create and activate a virtual environment (recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the package in development mode**:
   ```bash
   pip install -e .
   ```

4. **Install additional dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Getting Started

After installation, you can import and use the models directly in your Python code:

```python
from nio_jma import SNOHelmConv, NIOHelmPermInv, NIOHelmPermInvAbl

# Create a model instance
model = SNOHelmConv(
    input_dimensions_branch=100,
    input_dimensions_trunk=2,
    network_properties_branch={"n_basis": 64, "n_layers": 4, "act": "relu"},
    network_properties_trunk={"n_basis": 64, "n_layers": 4, "act": "relu"},
    fno_architecture={"modes1": 12, "width": 20, "n_layers": 4},
    device="cuda" if torch.cuda.is_available() else "cpu"
)
```

## Available Models

The package provides several models for different PDE problems:

### Helmholtz Equation
- `SNOHelmConv`: Standard NIO for Helmholtz equation
- `NIOHelmPermInv`: NIO with permutation invariance
- `NIOHelmPermInvAbl`: Ablation study variant

### Wave Equation
- `SNOWaveConv2`: Standard NIO for wave equation
- `NIOWavePerm`: NIO with permutation invariance
- `NIOWavePermAbl`: Ablation study variant

### Radiative Transfer
- `SNOConvRad`: Standard NIO for radiative transfer
- `NIORadPerm`: NIO with permutation invariance
- `NIORadPermAbl`: Ablation study variant

### Medical Imaging (EIT)
- `NIOHeartPerm`: NIO for EIT with heart and lungs
- `NIOHeartPermAbl`: Ablation study variant
- `SNOConvEIT`: Standard NIO for EIT

## Datasets

### Pre-computed Datasets
We provide pre-computed datasets for various PDE problems:

1. **Poisson, Helmholtz, and Radiative Transport Equations**
   - Download from: [Zenodo](https://zenodo.org/record/7566430) (14GB)
   - Or use the download script:
     ```bash
     python data/utils/download_data.py
     ```

2. **Seismic Imaging (OpenFWI Dataset)**
   - Download from: [OpenFWI](https://openfwi-lanl.github.io/docs/data.html#vel)
   - Preprocess the data:
     ```bash
     python data/utils/GetStyleData.py
     python data/utils/GetCurveData.py
     ```

## Training

Train models using the provided training script:

```bash
python -m training.scripts.RunNio OUTPUT_DIR PROBLEM MODEL [--workers N]
```

### Arguments:
- `OUTPUT_DIR`: Directory to save training results
- `PROBLEM`: Problem type (see below)
- `MODEL`: Model architecture (see below)
- `--workers`: Number of data loading workers (default: 0)

### Problem Types:
- `sine`: Caldéron problem with trigonometric coefficients
- `eit`: Caldéron problem with Heart&Lungs
- `helm`: Inverse wave scattering
- `rad`: Radiative transfer problem
- `curve`: Seismic imaging with CurveVel-A dataset
- `style`: Seismic imaging with Style-A dataset

### Model Architectures:
- `nio_new`: Neural Inverse Operator (NIO)
- `fcnn`: Fully Convolutional Neural Network
- `don`: DeepONet

### Example:
```bash
python -m training.scripts.RunNio results/helm_nio helm nio_new --workers 4
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{
  title={Neural Inverse Operators for Solving PDE Inverse Problems},
  author={Your Name and Coauthors},
  booktitle={Conference on Neural Information Processing Systems},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```

The models' hyperparameter can be specified in the corresponding python scripts as well.
To train the InversionNet model (the fully convolutional network baseline for Seismic Imaging) please refer to the GitHub page of Deng et Al (https://arxiv.org/pdf/2111.02926.pdf): https://github.com/lanl/OpenFWI

#### Hyperparameters Grid/Random Search
Cross validation for each model can be run with:

```
python3 ModelSelectionNIO.py model which
```

`which` and `model` must be one of the problems and models above.
For examples 
```
python3 ModelSelectionNIO.py nio_new helm
```
For the Seismic Imaging problem, only NIO and DON models can be run.

The hyperparameters of the models in the Table 1 have been obtained in this way.

The best performing configuration can be obtained by visualizing the results with tensorboard:
```
tensorboard --logdir=NameOfTheModelSelectionFolder
```

If a SLURM remote server is available set `sbatch=True` and `cluster="true"` in the script.

#### Pretrained Models
The models trained and used to compute the errors in Table 1 can be downloaded (9GB) by running:
```
python3 download_models.py
```
*Remark*: the compressed folder has to be unzipped!

#### Error Computations
The errors of the best performing models (Table 1) can be computed by running the script `ComputeNoiseErrors.py`.
It will compute the testing error for all the benchmark, for all the models and for different noise levels (Table 6).




