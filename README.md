# Age Detection using CNN

This project implements a basic Convolutional Neural Network (CNN) model for age detection using the IMDB-WIKI dataset. The model predicts the age from facial images. The project also includes a graphical user interface (GUI) for users to upload images and get the predicted age.

## Dataset

The dataset used for this project is the IMDB-WIKI dataset, which consists of the following files:
- `age.npy`: Contains the age labels.
- `gender.npy`: Contains the gender labels.
- `imdb-wiki-image-data.npy`: Contains the image data in NumPy array format.

Additionally, a pre-trained face detection model (`haarcascade_frontalface_default.xml`) is used to extract facial regions from the input images during prediction.

## Project Structure

```bash
├── Dataset
│ ├── imdb-wiki-image-npy
│ │ ├── age.npy
│ │ ├── gender.npy
│ │ ├── imdb-wiki-image-data.npy
│ └── haarcascade_frontalface_default.xml
├── model
│  └── model_L80_M6_mape24.h5
├── main.py
├── inference.py
├── gui.py
└── README.md
```


## Training

The model is trained on the IMDB-WIKI dataset with the following specifications:
- **Training Data**: 29,000 samples
- **Validation Data**: Remaining samples
- **Epochs**: 10
- **Batch Size**: 32


## GUI Application

A Tkinter-based GUI application allows users to upload an image and receive the predicted age. The GUI includes the following features:
- Uploading an image
- Displaying the uploaded image with rounded corners
- Displaying the predicted age

## Setup Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/zafar-TechWizard/Age-Gender-Detection.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Age-Gender-Detection
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the training script:
    ```bash
    python train_model.py
    ```

5. Run the GUI application:
    ```bash
    python gui.py
    ```

## Files

- `train_model.py`: Script for training the CNN model.
- `inference.py`: Script for predicting age from an input image
