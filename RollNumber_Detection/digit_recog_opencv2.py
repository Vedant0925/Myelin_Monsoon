import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense


mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


train_images, test_images = train_images / 255.0, test_images / 255.0


model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten 28x28 images to a 1D vector
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # Output layer with 10 classes (digits 0-9)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)


roll_number_image = cv2.imread('/Users/vedant/Downloads/WhatsApp Image 2023-08-18 at 14.04.31.jpeg', cv2.IMREAD_GRAYSCALE) #replace with custom img path
_, thresholded = cv2.threshold(roll_number_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


recognized_digits = []


def recognize_digit(contour):
    x, y, w, h = cv2.boundingRect(contour)
    
    if 10 < w < 100 and 20 < h < 120:
        digit_roi = roll_number_image[y:y+h, x:x+w]
        resized_digit = cv2.resize(digit_roi, (28, 28))  # Resize to match MNIST image size
        preprocessed_digit = resized_digit / 255.0
        
        
        predicted_digit = np.argmax(model.predict(np.array([preprocessed_digit])))
        
        recognized_digits.append((predicted_digit, (x, y)))


for contour in contours:
    recognize_digit(contour)


for digit, _ in recognized_digits:
    print("Recognized Digit: {}".format(digit))


for digit, (x, y) in recognized_digits:
    cv2.putText(roll_number_image, str(digit), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


cv2.imshow('Recognized Digits', roll_number_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
