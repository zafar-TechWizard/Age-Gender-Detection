import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

# Load the trained model
loaded_model = load_model("./Task 1/model_L80_M6_mape24.h5")

# Function to extract facial region from the image using a pre-trained face detection model
def extract_facial_region(image):
    # Load pre-trained face detection model (for example, Haar Cascade classifier)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Extract the largest face (assuming it's the main subject)
    if len(faces) > 0:
        (x, y, w, h) = max(faces, key=lambda face: face[2] * face[3])
        facial_region = image[y:y+h, x:x+w]
        return facial_region
    else:
        return None

# Function to handle button click event for uploading an image
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Load the image
        image = cv2.imread(file_path)

        # Extract facial region from the image
        facial_region = extract_facial_region(image)

        if facial_region is not None:
            # Resize the facial region to 100x100
            facial_region_resized = cv2.resize(facial_region, (100, 100))

            # Convert the facial region to grayscale
            facial_region_gray = cv2.cvtColor(facial_region_resized, cv2.COLOR_BGR2GRAY)

            # Reshape the facial region to match the input shape of the model (add a batch dimension and channel dimension)
            facial_region_input = np.expand_dims(facial_region_gray, axis=0)
            facial_region_input = np.expand_dims(facial_region_input, axis=3)  # Add channel dimension

            # Normalize the pixel values to the range [0, 1]
            facial_region_input = facial_region_input.astype('float32') / 255.0

            # Make predictions with the loaded model
            predicted_age = int(loaded_model.predict(facial_region_input)[0][0])

            # Display the uploaded image on the GUI with rounded corners
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image.thumbnail((250, 250))  # Resize image for display
            image = ImageTk.PhotoImage(image)

            # Clear previous image if exists
            if hasattr(image_label, 'photo'):
                image_label.photo = None

            image_label.config(image=image)
            image_label.image = image

            # Display the predicted age on the GUI
            age_label.config(text=f"Predicted Age: {predicted_age} years", font=("Roboto", 16))
        else:
            messagebox.showwarning("Warning", "No faces detected in the uploaded image.")
            # Clear the uploaded image and predicted age display
            clear_image()
    else:
        # Clear the uploaded image and predicted age display
        clear_image()

# Function to clear the uploaded image and predicted age display
def clear_image():
    image_label.config(image="")
    age_label.config(text="")

# Create the Tkinter window
window = tk.Tk()
window.title("Age Prediction")
window.geometry("320x480")
window.resizable(False, False)  # Disable window resizing

# Create a frame for the header with a cool background color
header_frame = tk.Frame(window, bg="#303F9F")
header_frame.pack(fill=tk.X)

# Create a label for the title in the header
title_label = tk.Label(header_frame, text="Age Prediction", font=("Roboto", 24, "bold"), bg="#303F9F", fg="white", padx=10, pady=10)
title_label.pack(side=tk.LEFT)

# Create a frame for the image display
image_frame = tk.Frame(window)
image_frame.pack(pady=20)

# Create a label for displaying the uploaded image with rounded corners
image_label = tk.Label(image_frame, bg="#EEEEEE")
image_label.pack(pady=10)

# Create a frame for the age prediction
prediction_frame = tk.Frame(window, bg="#EEEEEE")
prediction_frame.pack()

# Create a button to upload an image with improved UI/UX
upload_button = tk.Button(prediction_frame, text="Upload Image", command=upload_image, font=("Roboto", 14), bg="#2196F3", fg="white", bd=0, relief="raised", padx=10, pady=5, activebackground="#64B5F6", activeforeground="white", highlightthickness=0, borderwidth=0, highlightbackground="#303F9F")
upload_button.pack(pady=10)

# Create a label for displaying the predicted age
age_label = tk.Label(prediction_frame, text="", font=("Roboto", 16), bg="#EEEEEE")
age_label.pack(pady=10)

# Run the Tkinter event loop
window.mainloop()
