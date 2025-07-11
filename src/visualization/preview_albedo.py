import h5py
import os
import matplotlib.pyplot as plt

file_data = "data/Albedo.h5"
output_dir = "src/visualization/images"
os.makedirs(output_dir, exist_ok=True)

with h5py.File(file_data, 'r') as data:
    group_names = [key for key in data.keys() if isinstance(data[key], h5py.Group)]
    print(f"Number of groups (samples): {len(group_names)}")
    for i, group_name in enumerate(group_names):
        print(f"{i+1}. {group_name}")
    if not group_names:
        print("No groups found in the HDF5 file.")

    # Drill into first 5 samples of testing, training, and validation
    for split in ["testing", "training", "validation"]:
        if split in data:
            print(f"\nFirst 5 samples in '{split}':")
            split_group = data[split]
            sample_names = list(split_group.keys())
            for i, sample_name in enumerate(sample_names[:5]):
                print(f"  {i+1}. {sample_name}")
                sample = split_group[sample_name]
                for key in sample.keys():
                    obj = sample[key]
                    if isinstance(obj, h5py.Dataset):
                        print(f"      Dataset: {key}, shape: {obj.shape}, dtype: {obj.dtype}")
                    elif isinstance(obj, h5py.Group):
                        print(f"      Group: {key} (contains {len(obj.keys())} keys)")
        else:
            print(f"\nGroup '{split}' not found in the file.")

    # Save the first 5 input and output samples in training as images in a grid
    if "training" in data:
        training_group = data["training"]
        sample_names = list(training_group.keys())[:5]
        n_samples = len(sample_names)
        fig, axes = plt.subplots(n_samples, 2, figsize=(10, 2.5 * n_samples))
        for i, sample_name in enumerate(sample_names):
            sample = training_group[sample_name]
            input_img = sample["input"][:]
            output_vec = sample["output"][:]
            ax_input = axes[i, 0] if n_samples > 1 else axes[0]
            ax_output = axes[i, 1] if n_samples > 1 else axes[1]
            # Input as image
            im0 = ax_input.imshow(input_img, aspect='auto', cmap='viridis')
            ax_input.set_title(f"{sample_name} - input")
            fig.colorbar(im0, ax=ax_input, fraction=0.046, pad=0.04)
            # Output as line plot
            ax_output.plot(output_vec)
            ax_output.set_title(f"{sample_name} - output (line)")
            ax_output.set_xlabel("Index")
            ax_output.set_ylabel("Value")
        plt.tight_layout()
        out_path = os.path.join(output_dir, "albedo_training_first5_grid.png")
        plt.savefig(out_path)
        plt.close()
        print(f"\nSaved grid of first 5 training samples to {out_path}")
    else:
        print("'training' group not found in the file.")