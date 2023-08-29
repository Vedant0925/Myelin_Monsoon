import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from google.colab import files
import cv2


def upload_files():
    uploaded = files.upload()
    for filename in uploaded.keys():
        print(f'Uploaded file "{filename}" with length {len(uploaded[filename])} bytes')
    return list(uploaded.keys())

uploaded_image_files = upload_files()

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (_, _) = mnist.load_data()

train_images = train_images / 255.0

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

datagen = ImageDataGenerator(rotation_range=10,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.1)

train_images = train_images.reshape(-1, 28, 28, 1)

model.fit(datagen.flow(train_images, train_labels, batch_size=64),
          steps_per_epoch=len(train_images) / 64,
          epochs=15,
          verbose=2)

uploaded_image_path = uploaded_image_files[0]
uploaded_image = cv2.imread(uploaded_image_path, cv2.IMREAD_GRAYSCALE)
uploaded_image = cv2.resize(uploaded_image, (28, 28))
uploaded_image = uploaded_image / 255.0
uploaded_image = uploaded_image.reshape(1, 28, 28, 1)

predicted_digit = np.argmax(model.predict(uploaded_image))

plt.imshow(uploaded_image.reshape(28, 28), cmap='gray')
plt.title(f'Recognized Digit: {predicted_digit}', fontsize=16)
plt.axis('off')
plt.show()
