Brain Tumor Detection using MRI and Deep Learning

**NeuroSeg** is a deep learning-based **Brain Tumor Segmentation** system that analyzes MRI scans and highlights tumor regions. The project uses **U-Net** for segmentation and a **Flask backend** for processing, with a clean **frontend interface** to upload and visualize results.

🚀 **Live Demo:** *(Coming Soon after deployment)*  
📂 **Dataset Used:** [LGG Segmentation Dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-segmentation)  
🔗 **GitHub Repo:** [NeuroSeg](https://github.com/mystichronicle/NeuroSeg)  

---

## **📌 Features**
✔ **MRI Scan Upload:** Users can upload an MRI scan of the brain.  
✔ **AI-Based Segmentation:** The model detects tumor regions in the image.  
✔ **Processed Image Output:** The result is displayed with an overlay on the original image.  
✔ **Fast & Accurate:** Uses **U-Net** for high-precision segmentation.  
✔ **Web-Based Interface:** Simple **frontend UI** with **drag & drop upload.**  
✔ **Fully Deployed:** Works online via **Flask (Backend) + GitHub Pages (Frontend).**  

---

## **📌 Technologies Used**
### **Backend (API)**
- 🐍 **Python 3.10**
- ⚙ **Flask** (for web API)
- 🔬 **TensorFlow/Keras** (for deep learning)
- 🖼 **OpenCV** (for image processing)
- 🗂 **NumPy, Matplotlib** (for visualization)
- 🛠 **Gunicorn** (for deployment)

### **Frontend (Web App)**
- 🌐 **HTML, CSS, JavaScript**
- 🎨 **Modern UI with CSS Flexbox/Grid**
- ⚡ **GitHub Pages for free hosting**

---

## **📌 Installation & Setup**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/mystichronicle/NeuroSeg.git
cd NeuroSeg
```

### **2️⃣ Set Up Python Environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### **3️⃣ Run Flask API Locally**
```bash
cd backend
python app.py
```
🚀 API will be running at **http://127.0.0.1:5580/**

### **4️⃣ Open Frontend**
Open **frontend/index.html** in your browser.

---

## **📌 Usage Guide**
1️⃣ Upload an MRI scan (.png/.jpg/.tif).  
2️⃣ Click **"Upload & Analyze"**.  
3️⃣ The AI will process and highlight tumor regions.  
4️⃣ The processed image will be displayed.  

---

## **📌 Deployment Steps**
### **Backend (Flask API) - Hosted on Render**
1️⃣ Push code to GitHub  
2️⃣ Deploy Flask API using Render  
3️⃣ Set start command:
```bash
gunicorn --bind 0.0.0.0:10000 backend.app:app
```
4️⃣ Get live API URL (e.g., **https://neuroseg.onrender.com**)

### **Frontend (Web App) - Hosted on GitHub Pages**
1️⃣ Move frontend files to **/frontend/**  
2️⃣ Push frontend to GitHub  
3️⃣ Enable **GitHub Pages** (Settings → Pages → Select main)  
4️⃣ The frontend will be live at:  
```bash
https://mystichronicle.github.io/NeuroSeg-Frontend/
```

---

## **📌 File Structure**
```
NeuroSeg/
│── backend/              # Flask Backend
│   ├── app.py            # Flask main app
│   ├── models/           # Deep Learning models
│   ├── preprocessing/    # Data processing scripts
│   ├── templates/        # HTML frontend for Flask
│   ├── static/           # Static CSS & JS
│   ├── uploads/          # Stores uploaded MRI scans
│── frontend/             # Standalone Web App (GitHub Pages)
│   ├── index.html        # Main Web Page
│   ├── static/           # JS, CSS, Images
│── data/                 # Dataset (MRI Scans)
│── models/               # Trained models (.h5 files)
│── README.md             # Project Documentation
│── requirements.txt      # Python dependencies
```


---


🤝 Want to contribute? Open a **Pull Request!**  

---

## **📌 License**
This project is open-source under the **MIT License**.  
Feel free to use, modify, and share! 🚀  

---

## **📌 Acknowledgments**
🧠 Inspired by the **LGG Segmentation Dataset**  
📄 **Citations:**
- Mateusz Buda, et al. *"Radiogenomics of lower-grade glioma"*
- *U-Net: Convolutional Networks for Biomedical Image Segmentation*
