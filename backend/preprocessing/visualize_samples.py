import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend

import matplotlib.pyplot as plt
import cv2
import os

# Define absolute paths
MRI_PATH = "/home/debjit/Programming/ML/BrainTumorSegmentation/data/processed/mri_scans/"
MASK_PATH = "/home/debjit/Programming/ML/BrainTumorSegmentation/data/processed/tumor_masks/"

# Get list of saved images
mri_files = sorted(os.listdir(MRI_PATH))[:5]
mask_files = sorted(os.listdir(MASK_PATH))[:5]

# Plot images
fig, axes = plt.subplots(5, 2, figsize=(10, 15))

for i, (mri_file, mask_file) in enumerate(zip(mri_files, mask_files)):
    mri = cv2.imread(os.path.join(MRI_PATH, mri_file), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(os.path.join(MASK_PATH, mask_file), cv2.IMREAD_GRAYSCALE)

    axes[i, 0].imshow(mri, cmap="gray")
    axes[i, 0].set_title(f"MRI Scan: {mri_file}")

    axes[i, 1].imshow(mask, cmap="Reds")
    axes[i, 1].set_title(f"Tumor Mask: {mask_file}")

plt.tight_layout()

# Save the plot instead of showing it
plt.savefig("sample_visualization.png")
print("✅ Visualization saved as 'sample_visualization.png'")
