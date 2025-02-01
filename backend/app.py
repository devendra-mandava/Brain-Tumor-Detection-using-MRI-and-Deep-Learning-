import os
import numpy as np
import cv2
import gdown
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

DATA_FOLDER_ID = "1IvKCI67sYBxZZZOPz5YPRJIFpWVrRdjf"
MODEL_FILE_ID = "1yafZdUtwbhFIy2fDFtzkMj3ld5ZINLTV"

BASE_DIR = os.getcwd()
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads/")
MODEL_PATH = os.path.join(BASE_DIR, "models/unet_brain_segmentation.h5")
DATA_PATH = os.path.join(BASE_DIR, "data/")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def download_data():
    if not os.path.exists(DATA_PATH) or len(os.listdir(DATA_PATH)) == 0:
        print("🔽 Data folder not found! Downloading from Google Drive...")
        gdown.download_folder(
            f"https://drive.google.com/drive/folders/{DATA_FOLDER_ID}", output=DATA_PATH, quiet=False)
        print("✅ Data folder downloaded successfully!")


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("🔽 Model file not found! Downloading from Google Drive...")
        gdown.download(
            f"https://drive.google.com/uc?id={MODEL_FILE_ID}", MODEL_PATH, quiet=False)
        print("✅ Model downloaded successfully!")


download_data()
download_model()

print("📥 Loading trained UNet model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        output_path = segment_brain_tumor(file_path)

        return jsonify({"message": "File uploaded successfully", "output_image": output_path})

    return jsonify({"error": "Invalid file format"}), 400


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


def segment_brain_tumor(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original_shape = img.shape
    img_resized = cv2.resize(img, (256, 256)) / 255.0
    img_resized = np.expand_dims(img_resized, axis=(0, -1))

    predicted_mask = model.predict(img_resized)[0]
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8)

    predicted_mask_resized = cv2.resize(
        predicted_mask, (original_shape[1], original_shape[0]))

    print(f"🔍 Predicted Mask Sum: {np.sum(predicted_mask_resized)}")

    if np.sum(predicted_mask_resized) == 0:
        print("⚠️ Warning: Model output is empty! Adjusting contrast...")
        predicted_mask_resized = predicted_mask_resized.astype(
            np.float32)
        predicted_mask_resized += 0.2
        predicted_mask_resized = np.clip(
            predicted_mask_resized, 0, 1)
        predicted_mask_resized = (
            predicted_mask_resized * 255).astype(np.uint8)

    overlay = cv2.addWeighted(img, 0.7, predicted_mask_resized * 255, 0.3, 0)

    output_filename = "output.png"
    output_path = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)
    cv2.imwrite(output_path, overlay)

    if not os.path.exists(output_path):
        print("❌ ERROR: Output image was not saved!")
        return None

    print(f"✅ Output image saved at {output_path}")
    return f"/uploads/{output_filename}"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
