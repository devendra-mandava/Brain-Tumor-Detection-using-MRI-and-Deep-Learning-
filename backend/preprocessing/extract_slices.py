import os
import cv2
import numpy as np
from glob import glob

RAW_DATA_PATH = "/home/debjit/Programming/ML/BrainTumorSegmentation/data/raw/LGG-MRI-Segmentation/"
PROCESSED_DATA_PATH = "/home/debjit/Programming/ML/BrainTumorSegmentation/data/processed/"

os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DATA_PATH, "mri_scans"), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DATA_PATH, "tumor_masks"), exist_ok=True)

patient_folders = sorted(glob(os.path.join(RAW_DATA_PATH, "*")))

processed_count = 0
for patient_folder in patient_folders:
    if not os.path.isdir(patient_folder):
        continue  
    
    mri_files = sorted(glob(os.path.join(patient_folder, "*.tif")))
    mri_files = [f for f in mri_files if "_mask" not in f]  

    for mri_path in mri_files:
        mask_path = mri_path.replace(".tif", "_mask.tif")  
        
        if not os.path.exists(mask_path):
            print(f"⚠️ No mask found for {mri_path}, skipping...")
            continue  
        
        try:
            mri = cv2.imread(mri_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            mri_resized = cv2.resize(mri, (256, 256))
            mask_resized = cv2.resize(mask, (256, 256))

            cv2.imwrite(os.path.join(PROCESSED_DATA_PATH, "mri_scans", f"{processed_count}.png"), mri_resized)
            cv2.imwrite(os.path.join(PROCESSED_DATA_PATH, "tumor_masks", f"{processed_count}.png"), mask_resized)

            processed_count += 1
            print(f"✅ Processed {mri_path}")

        except Exception as e:
            print(f"❌ Error processing {mri_path}: {e}")

print(f"✅ MRI scan extraction complete! Total processed: {processed_count}")
print(f"📂 Check processed images in: {PROCESSED_DATA_PATH}")
