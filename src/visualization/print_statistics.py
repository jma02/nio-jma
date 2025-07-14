import h5py
import numpy as np

file_path = "data/HelmholtzTomography.h5"

with h5py.File(file_path, "r") as hdf:
    for key in hdf.keys():
        obj = hdf[key]
        if isinstance(obj, h5py.Group):
            print(f"Group: {key}")
        elif isinstance(obj, h5py.Dataset):
            print(f"Dataset: {key}, shape: {obj.shape}, dtype: {obj.dtype}")

with h5py.File(file_path, "r") as hdf:
    if "grid" in hdf:
        grid = hdf["grid"][:]
        print(f"grid shape: {grid.shape}, dtype: {grid.dtype}")
        # Pretty print the first and second channels (shape: 70, 70, 2)
        np.set_printoptions(precision=3, suppress=True, linewidth=120)
        print("First channel of grid:")
        print(grid[:, :, 0])
        print("Second channel of grid:")
        print(grid[:, :, 1])
    else:
        print("No 'grid' dataset found in the file.")