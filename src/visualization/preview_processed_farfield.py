import h5py

file_path = "data/merged_data_split.hdf5"

def print_h5_structure(name, obj):
    indent = '  ' * (name.count('/') - 1)
    if isinstance(obj, h5py.Group):
        print(f"{indent}Group: {name.split('/')[-1]}")
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}Dataset: {name.split('/')[-1]}, shape: {obj.shape}, dtype: {obj.dtype}")

with h5py.File(file_path, "r") as hdf:
    hdf.visititems(print_h5_structure)