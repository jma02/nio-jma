import h5py
import numpy as np

farf_file_path = "data/merged_data.hdf5"
output_file = "data/merged_data_split.hdf5"


# Load data
with h5py.File(farf_file_path, "r") as hdf_file:
    image = hdf_file["image"][:]
    farfield_real = hdf_file["farfield.real"][:]
    farfield_imag = hdf_file["farfield.imag"][:]



N_TEST_SAMPLES = farfield_real.shape[1]
print(f"Number of test samples: {N_TEST_SAMPLES}")

# Reshape data
images = image.T.reshape(N_TEST_SAMPLES, 100, 100)
f_r = farfield_real.T.reshape(N_TEST_SAMPLES, 100, 100)
f_i = farfield_imag.T.reshape(N_TEST_SAMPLES, 100, 100)
