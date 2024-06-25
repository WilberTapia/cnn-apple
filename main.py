import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import keras.src.utils as image
from keras.src import layers 
from keras.src.applications.vgg16 import VGG16
from keras.src.models import Model
from keras.api.saving import load_model
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import numpy as np
# from keras.src.models.model import load_weights

def predict_apple(img_path):
  """
  Predicts whether an image contains an apple using a pre-trained VGG16 model.

  Args:
      img_path (str): Path to the image file.

  Returns:
      bool: True if the image is predicted to contain an apple, False otherwise.
  """

  # # Load the pre-trained VGG16 model without the top layers (freeze weights)
  # base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
  # for layer in base_model.layers:
  #   layer.trainable = False

  # # Add custom layers for classification
  # x = base_model.output
  # x = layers.Flatten()(x)
  # x = layers.Dense(512, activation='relu')(x)
  # x = layers.Dropout(0.5)(x)
  # predictions = layers.Dense(1, activation='sigmoid')(x)

  # # Create the final model
  # model = Model(inputs=base_model.input, outputs=predictions)
  # # Create a new model instance
  # model.load_weights('apple_classifier.weights.h5')

  model = load_model('apple_classifier.keras')
  # Load the model weights (assuming they are saved after training)
  # model =  load_weights('apple_classifier.keras','apple_classifier.keras', skip_mismatch=False)  # Replace with your weights file path

  # Preprocess the image
  img = image.load_img(img_path, target_size=(224, 224))
  x = image.img_to_array(img)
  x = x / 255.0  # Rescale to [0, 1]
  # Add a batch dimension
  x = np.expand_dims(x, axis=0)

  # Make prediction
  prediction = model.predict(x)[0][0]
  apple_probability = prediction

  # Set a threshold for classification (adjust as needed)
  threshold = 0.5  # Consider image an apple if probability >= threshold
  print('Apple probability:', apple_probability)

  return apple_probability >= threshold

# Example usage
if __name__ == '__main__':
    print(predict_apple('man.jpg'))
    print(predict_apple('manz.jpg'))
    print(predict_apple('img1.jpg'))
    print(predict_apple('img2.jpg'))
    print(predict_apple('img3.jpg'))

