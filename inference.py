import cv2
import numpy as np
from tensorflow.keras.models import load_model




def extract_facial_region(image):
    # Load pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # image to grayscale
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
    


def detect(img):

    model = load_model("./model/model_L80_M6_mape24.h5")

    sample_image = cv2.imread(img)

    facial_region = extract_facial_region(sample_image)

    if facial_region is not None:
        facial_region_resized = cv2.resize(facial_region, (100, 100))

        facial_region_gray = cv2.cvtColor(facial_region_resized, cv2.COLOR_BGR2GRAY)

        facial_region_input = np.expand_dims(facial_region_gray, axis=0)
        facial_region_input = np.expand_dims(facial_region_input, axis=3)

        # Normalize the pixel values to the range [0, 1]
        facial_region_input = facial_region_input.astype('float32') / 255.0

        # Make predictions with the loaded model
        predicted_age = int(model.predict(facial_region_input))

        return predicted_age
    else:
        return "No faces detected in the sample image."
    
if __name__ == "__main__":
    image = " " # add image path
    prediction = detect(image)
    print("Predicted age: ")