import h5py
import numpy as np

farf_file_path = "data/merged_data.hdf5"
output_file = "data/merged_data_split.hdf5"


# Load data
with h5py.File(farf_file_path, "r") as hdf_file:
    image = hdf_file["image"][:]
    farfield_real = hdf_file["farfield.real"][:]
    farfield_imag = hdf_file["farfield.imag"][:]



N_SAMPLES = farfield_real.shape[1]
print(f"Number of test samples: {N_SAMPLES}")

# Reshape data
images = image.T.reshape(N_SAMPLES, 100, 100)
f_r = farfield_real.T.reshape(N_SAMPLES, 100, 100)
f_i = farfield_imag.T.reshape(N_SAMPLES, 100, 100)

# Make grid
grid_x = np.tile(np.linspace(0, 1, 100), (100, 1))
grid_y = np.tile(np.linspace(0, 1, 100), (100, 1)).T

max_inp_re = np.max(f_r)
max_inp_im = np.max(f_i)
min_inp_re = np.min(f_r)
min_inp_im = np.min(f_i)

max_out = np.max(images)
min_out = np.min(images)

mean_inp_re = np.mean(f_r, axis=0)
mean_inp_im = np.mean(f_i, axis=0)

mean_out = np.mean(images, axis=0)

std_inp_re = np.std(f_r, axis=0)
std_inp_im = np.std(f_i, axis=0)
std_out = np.std(images, axis=0)

training_samples = 15000
validation_samples = 2500
testing_samples = N_SAMPLES - training_samples - validation_samples

# Save to new HDF5 file
with h5py.File(output_file, "w") as hdf_out:
    # Create grid dataset (combine grid_x and grid_y)
    grid = np.stack([grid_x, grid_y], axis=-1)
    hdf_out.create_dataset("grid", data=grid)
    
    # Create scalar datasets
    hdf_out.create_dataset("max_inp_real", data=max_inp_re)
    hdf_out.create_dataset("max_inp_imag", data=max_inp_im)
    hdf_out.create_dataset("max_out", data=max_out)
    hdf_out.create_dataset("min_inp_real", data=min_inp_re)
    hdf_out.create_dataset("min_inp_imag", data=min_inp_im)
    hdf_out.create_dataset("min_out", data=min_out)
    
    # Create mean and std datasets
    hdf_out.create_dataset("mean_inp_fun_real", data=mean_inp_re)
    hdf_out.create_dataset("mean_inp_fun_imag", data=mean_inp_im)
    hdf_out.create_dataset("mean_out_fun", data=mean_out)
    hdf_out.create_dataset("std_inp_fun_real", data=std_inp_re)
    hdf_out.create_dataset("std_inp_fun_imag", data=std_inp_im)
    hdf_out.create_dataset("std_out_fun", data=std_out)
    
    # Create groups for data splits
    training_group = hdf_out.create_group("training")
    validation_group = hdf_out.create_group("validation")
    testing_group = hdf_out.create_group("testing")
    
    # Split and save data
    train_end = training_samples
    val_end = train_end + validation_samples
    
    # Training data
    training_group.create_dataset("input_real", data=f_r[:train_end])
    training_group.create_dataset("input_imag", data=f_i[:train_end])
    training_group.create_dataset("output", data=images[:train_end])
    
    # Validation data
    validation_group.create_dataset("input_real", data=f_r[train_end:val_end])
    validation_group.create_dataset("input_imag", data=f_i[train_end:val_end])
    validation_group.create_dataset("output", data=images[train_end:val_end])
    
    # Testing data
    testing_group.create_dataset("input_real", data=f_r[val_end:])
    testing_group.create_dataset("input_imag", data=f_i[val_end:])
    testing_group.create_dataset("output", data=images[val_end:])