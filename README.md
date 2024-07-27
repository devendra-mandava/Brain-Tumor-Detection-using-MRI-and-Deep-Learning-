
# Brain Tumor Detection Using MRI Images

This project aims to detect brain tumors from MRI images using a Convolutional Neural Network (CNN). The dataset contains MRI images labeled as either having a brain tumor (`yes`) or not having a brain tumor (`no`). The project involves preprocessing the images, training a CNN, evaluating its performance, and visualizing the results.

## Table of Contents

- [Brain Tumor Detection Using MRI Images](#brain-tumor-detection-using-mri-images)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Dataset](#dataset)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Results](#results)
  - [Contributing](#contributing)
  - [License](#license)

## Introduction

Brain tumors are abnormal growths of cells in the brain that can be benign or malignant. Detecting brain tumors early is crucial for effective treatment. This project uses deep learning techniques to automate the detection of brain tumors from MRI images.

## Dataset

The dataset used in this project contains MRI images labeled as either `no` (no brain tumor) or `yes` (brain tumor). The dataset is organized into two directories:

- `no`: Images without brain tumors.
- `yes`: Images with brain tumors.

This dataset can be found in **Kaggle**: [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

## Project Structure

The project has the following structure:

```
Brain-Tumor-Detection/
├── brain_mri_images/
│   ├── no/
│   ├── yes/
├── Brain_MRI_Analysis.ipynb
├── README.md
└── requirements.txt
```

- `brain_mri_images/`: Contains the MRI images dataset.
- `Brain_MRI_Analysis.ipynb`: Jupyter Notebook with the full analysis and model training.
- `README.md`: Project documentation.
- `requirements.txt`: List of dependencies required for the project.

## Installation

To run this project, you need to have Python and Jupyter Notebook installed. You can install the required dependencies using the following command:

```sh
pip install -r requirements.txt
```

The `requirements.txt` file contains:

```
numpy
matplotlib
seaborn
tensorflow
scikit-learn
opencv-python
```

## Usage

1. **Clone the repository:**

```sh
git clone https://github.com/debjit-mandal/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection
```

2. **Download the dataset:**

Place the `no` and `yes` directories containing the MRI images into the `brain_mri_images/` directory.

3. **Run the Jupyter Notebook:**

```sh
jupyter notebook Brain_MRI_Analysis.ipynb
```

Follow the steps in the notebook to preprocess the data, train the model, and evaluate its performance.

## Results

The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score. The notebook includes visualizations of the training process and the results, including confusion matrices and plots of accuracy and loss over epochs.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please create an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
