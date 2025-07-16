I've removed the .gitattributes from this folder, so upsptream changes will not be tracked.

Use `wget` to download the .pt files
[https://huggingface.co/AnshDesai1/Scattering/tree/main](https://huggingface.co/AnshDesai1/Scattering/tree/main)

For example
```
wget https://huggingface.co/AnshDesai1/Scattering/resolve/main/BCNN_tuned_model_kvalue1.pt
```
---
license: mit
---
# Inverse Scattering with Neural Networks

This repository contains code and pretrained models for evaluating and comparing different neural network architectures (CNN, BCNN, CNNB, FNO) in approximating inverse scattering maps from far-field data.

## Files and Descriptions

| File | Description |
|------|-------------|
| `farfield_image_test.pt` | Testing Dataset. |
| `farfield_improved.pt` | Training Dataset for FNO and CNNB. |
| `farfield_epsilon.pt` | Training Dataset for BCNN and CNN. |
| `fno.pt` | Trained FNO model. |
| `BCNN_tuned_model_kvalue1.pt` | Trained BCNN model. |
| `CNNB_tuned_model_kvalue1.pt` | Trained CNNB model. |
| `CNN_tuned_model_kvalue1.pt` | Trained CNN model. |
| `fno_tune.py` | Script for training/tuning the FNO model. |
| `tuning.sh` | Hyperparameter tuning script for cluster usage (sbatch). |
| `test.py` | Script for loading models, running inference on test data, and evaluating error metrics. |

## Data Generation
For data generation, see https://github.com/nibj/Helmholtz-scattering-data

## License

MIT License. See `LICENSE` file for details.
