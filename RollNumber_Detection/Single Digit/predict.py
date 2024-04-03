import numpy as np
from tensorflow import keras
from PIL import Image

def predict_handwritten_digit(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28), Image.ANTIALIAS)  # Resize to 28x28 pixels
    
    img_array = np.array(img)
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for model (1, 28, 28, 1)
    img_array = img_array.astype('float32') / 255.0  # Normalize pixel values
    
    model = keras.models.load_model('path to model')
    
    prediction = model.predict(img_array)
    digit = np.argmax(prediction)

    return digit

image_path = 'path to image'
digit = predict_handwritten_digit(image_path)
print(f'The predicted digit is: {digit}')
