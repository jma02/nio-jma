import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colormaps as cmaps
import cmocean as cmo

from pathlib import Path
import h5py

# Set plot styles
plt.rc('text', usetex=False)  # Disable LaTeX rendering
plt.rc('font', family='Nimbus Roman')

# Define colormap
colors = [(0.32, 0.31, 0.43), (0.54, 0.53, 0.69), (0.6, 0.78, 0.78), (1, 1, 1)]  # Dark blue, teal, white
cmap_name = 'custom_colormap'
cm_custom = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

# File paths
farf_file_path = "data/merged_data.hdf5"

# Load data
with h5py.File(farf_file_path, "r") as hdf_file:
    image = hdf_file["image"][:]
    farfield_real = hdf_file["farfield.real"][:]
    farfield_imag = hdf_file["farfield.imag"][:]



N_TEST_SAMPLES = farfield_real.shape[1]

# Reshape data
images = image.T.reshape(N_TEST_SAMPLES, 100, 100)
f_r = farfield_real.T.reshape(N_TEST_SAMPLES, 100, 100)
f_i = farfield_imag.T.reshape(N_TEST_SAMPLES, 100, 100)

# Select samples
i1 = images[9]+1
i2 = images[681]+1

f1r = f_r[9]
f1i = f_i[9]

f2r = f_r[681]
f2i = f_i[681]

# Set up figure
fig, axs = plt.subplots(2, 3, figsize=(14, 8))

cmap1 = 'cmo.dense'
cmap2 = 'cmo.balance'

# Plot first row
im1 = axs[0, 0].imshow(i1, extent=(-1, 1, -1, 1), origin='lower', cmap=cmap1, vmin=1, vmax=2)
axs[0, 0].set_xlabel(r'(I.) $\eta(\mathbf{x})$', fontsize=14)
axs[0, 0].set_xticks([-1, 0, 1])
axs[0, 0].set_xticklabels(['-1', '0', '1'])
axs[0, 0].set_yticks([-1, 0, 1])
axs[0, 0].set_yticklabels(['-1', '0', '1'])

im2 = axs[0, 1].imshow(f1r, extent=(0, 2 * np.pi, 0, 2 * np.pi), origin='lower', cmap=cmap2, vmin=min(f2r.min(),f1r.min()), vmax=max(f2r.max(),f1r.max()))
axs[0, 1].set_xlabel(r'(II.) $\Re(u_{\infty})$', fontsize=14)
axs[0, 1].set_xticks([0, np.pi, 2 * np.pi])
axs[0, 1].set_xticklabels(['$0$', r'$\pi$', r'$2\pi$'])
axs[0, 1].set_yticks([0, np.pi, 2 * np.pi])
axs[0, 1].set_yticklabels(['$0$', r'$\pi$', r'$2\pi$'])

im3 = axs[0, 2].imshow(f1i, extent=(0, 2 * np.pi, 0, 2 * np.pi), origin='lower', cmap=cmap2, vmin=min(f2i.min(),f1i.min()), vmax=max(f2i.max(),f1i.max()))
axs[0, 2].set_xlabel(r'(III.) $\Im(u_{\infty})$', fontsize=14)
axs[0, 2].set_xticks([0, np.pi, 2 * np.pi])
axs[0, 2].set_xticklabels(['$0$', r'$\pi$', r'$2\pi$'])
axs[0, 2].set_yticks([0, np.pi, 2 * np.pi])
axs[0, 2].set_yticklabels(['$0$', r'$\pi$', r'$2\pi$'])

# Plot second row
im4 = axs[1, 0].imshow(i2, extent=(-1, 1, -1, 1), origin='lower', cmap=cmap1, vmin=1, vmax=2)
axs[1, 0].set_xlabel(r'(IV.) $\eta(\mathbf{x})$', fontsize=14)
axs[1, 0].set_xticks([-1, 0, 1])
axs[1, 0].set_xticklabels(['-1', '0', '1'])
axs[1, 0].set_yticks([-1, 0, 1])
axs[1, 0].set_yticklabels(['-1', '0', '1'])

im5 = axs[1, 1].imshow(f2r, extent=(0, 2 * np.pi, 0, 2 * np.pi), origin='lower', cmap=cmap2, vmin=min(f2r.min(),f1r.min()), vmax=max(f2r.max(),f1r.max()))
axs[1, 1].set_xlabel(r'(V.) $\Re(u_{\infty})$', fontsize=14)
axs[1, 1].set_xticks([0, np.pi, 2 * np.pi])
axs[1, 1].set_xticklabels(['$0$', r'$\pi$', r'$2\pi$'])
axs[1, 1].set_yticks([0, np.pi, 2 * np.pi])
axs[1, 1].set_yticklabels(['$0$', r'$\pi$', r'$2\pi$'])

im6 = axs[1, 2].imshow(f2i, extent=(0, 2 * np.pi, 0, 2 * np.pi), origin='lower', cmap=cmap2, vmin=min(f2i.min(),f1i.min()), vmax=max(f2i.max(),f1i.max()))
axs[1, 2].set_xlabel(r'(VI.) $\Im(u_{\infty})$', fontsize=14)
axs[1, 2].set_xticks([0, np.pi, 2 * np.pi])
axs[1, 2].set_xticklabels(['$0$', r'$\pi$', r'$2\pi$'])
axs[1, 2].set_yticks([0, np.pi, 2 * np.pi])
axs[1, 2].set_yticklabels(['$0$', r'$\pi$', r'$2\pi$'])

fig.colorbar(im1, ax=axs[0, 0], fraction=0.046, pad=0.04)
fig.colorbar(im2, ax=axs[0, 1], fraction=0.046, pad=0.04)
fig.colorbar(im3, ax=axs[0, 2], fraction=0.046, pad=0.04)
fig.colorbar(im4, ax=axs[1, 0], fraction=0.046, pad=0.04)
fig.colorbar(im5, ax=axs[1, 1], fraction=0.046, pad=0.04)
fig.colorbar(im6, ax=axs[1, 2], fraction=0.046, pad=0.04)
# Save the figure
out_path = Path("src/visualization/images/preview_born_farfield.png")
plt.savefig(out_path)