import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

MODEL_PATH = "/home/debjit/Programming/ML/BrainTumorSegmentation/models/unet_brain_segmentation.h5"
TEST_IMAGE_PATH = "/home/debjit/Programming/ML/BrainTumorSegmentation/data/processed/mri_scans/"
TEST_MASK_PATH = "/home/debjit/Programming/ML/BrainTumorSegmentation/data/processed/tumor_masks/"

print("📥 Loading trained UNet model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

test_images = sorted(os.listdir(TEST_IMAGE_PATH))
test_masks = sorted(os.listdir(TEST_MASK_PATH))

idx = np.random.randint(0, len(test_images))
test_img_file = test_images[idx]
test_mask_file = test_masks[idx]

print(f"🖼️ Testing on image: {test_img_file}")

test_img = cv2.imread(os.path.join(TEST_IMAGE_PATH, test_img_file), cv2.IMREAD_GRAYSCALE)
test_mask = cv2.imread(os.path.join(TEST_MASK_PATH, test_mask_file), cv2.IMREAD_GRAYSCALE)

test_img_resized = cv2.resize(test_img, (256, 256)) / 255.0
test_img_resized = np.expand_dims(test_img_resized, axis=(0, -1))  

predicted_mask = model.predict(test_img_resized)[0]  
predicted_mask = (predicted_mask > 0.5).astype(np.uint8)  

predicted_mask_resized = cv2.resize(predicted_mask, (test_img.shape[1], test_img.shape[0]))

overlay = cv2.addWeighted(test_img, 0.7, predicted_mask_resized * 255, 0.3, 0)

fig, ax = plt.subplots(1, 4, figsize=(15, 5))

ax[0].imshow(test_img, cmap="gray")
ax[0].set_title("Original MRI")

ax[1].imshow(test_mask, cmap="gray")
ax[1].set_title("Ground Truth Mask")

ax[2].imshow(predicted_mask_resized, cmap="gray")
ax[2].set_title("Predicted Mask")

ax[3].imshow(overlay, cmap="gray")
ax[3].set_title("Overlay Prediction")

for a in ax:
    a.axis("off")

plt.show()
