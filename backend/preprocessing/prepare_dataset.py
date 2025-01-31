import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split

# Define dataset paths
PROCESSED_DATA_PATH = "/home/debjit/Programming/ML/BrainTumorSegmentation/data/processed/"
OUTPUT_PATH = "/home/debjit/Programming/ML/BrainTumorSegmentation/data/processed_numpy/"

# Ensure output directory exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Image size for UNet input
IMG_SIZE = (256, 256)
BATCH_SIZE = 500  # Process 500 images at a time to prevent memory overload

# Get list of MRI scans & tumor masks
mri_files = sorted(os.listdir(os.path.join(PROCESSED_DATA_PATH, "mri_scans")))
mask_files = sorted(os.listdir(os.path.join(PROCESSED_DATA_PATH, "tumor_masks")))

# Ensure MRI and mask sizes match
if len(mri_files) != len(mask_files):
    min_size = min(len(mri_files), len(mask_files))
    mri_files = mri_files[:min_size]
    mask_files = mask_files[:min_size]

print(f"📊 Total MRI Scans: {len(mri_files)}")
print(f"📊 Total Tumor Masks: {len(mask_files)}")

# Split dataset into training and validation sets first
train_mri_files, val_mri_files, train_mask_files, val_mask_files = train_test_split(
    mri_files, mask_files, test_size=0.2, random_state=42
)

# Function to process images in batches
def process_and_save(files_mri, files_mask, set_name):
    total = len(files_mri)
    for i in range(0, total, BATCH_SIZE):
        mri_batch = []
        mask_batch = []

        batch_files_mri = files_mri[i : i + BATCH_SIZE]
        batch_files_mask = files_mask[i : i + BATCH_SIZE]

        for mri_file, mask_file in zip(batch_files_mri, batch_files_mask):
            mri_path = os.path.join(PROCESSED_DATA_PATH, "mri_scans", mri_file)
            mask_path = os.path.join(PROCESSED_DATA_PATH, "tumor_masks", mask_file)

            try:
                mri = cv2.imread(mri_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                # Resize images to UNet input size
                mri_resized = cv2.resize(mri, IMG_SIZE) / 255.0  # Normalize to [0,1]
                mask_resized = cv2.resize(mask, IMG_SIZE) / 255.0  # Normalize to [0,1]

                mri_batch.append(mri_resized)
                mask_batch.append(mask_resized)

            except Exception as e:
                print(f"❌ Error processing {mri_file}: {e}")
                continue

        # Convert to numpy arrays and save in batches
        np.save(os.path.join(OUTPUT_PATH, f"X_{set_name}_{i}.npy"), np.array(mri_batch).reshape(-1, 256, 256, 1))
        np.save(os.path.join(OUTPUT_PATH, f"y_{set_name}_{i}.npy"), np.array(mask_batch).reshape(-1, 256, 256, 1))
        print(f"✅ Processed {i + len(mri_batch)} / {total} for {set_name} set")

# Process and save training set
print("\n🔹 Processing Training Set...")
process_and_save(train_mri_files, train_mask_files, "train")

# Process and save validation set
print("\n🔹 Processing Validation Set...")
process_and_save(val_mri_files, val_mask_files, "val")

print(f"✅ All data saved in {OUTPUT_PATH}")
