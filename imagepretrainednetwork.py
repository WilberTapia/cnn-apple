from  keras.src.legacy.preprocessing.image import ImageDataGenerator
import os 
import zipfile 
import tensorflow as tf 
from keras.src import layers 
from keras.src import Model 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.src.applications.vgg16 import VGG16
from keras.src.optimizers import RMSprop
from keras.src import Model

TF_ENABLE_ONEDNN_OPTS=0

local_zip = 'dataset/cqji12emccqmasafan2839.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('dataset/')
zip_ref.close()

base_dir = 'dataset/data'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')

# Directory with our training cat pictures
train_apple_dir = os.path.join(train_dir, 'apple')

# Directory with our validation cat pictures
validation_apple_dir = os.path.join(validation_dir, 'apple')

nrows = 4
ncols = 4

fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)
pic_index = 100
train_apple_fnames = os.listdir( train_apple_dir )


next_apple_pix = [os.path.join(train_apple_dir, fname) 
                for fname in train_apple_fnames[ pic_index-8:pic_index] 
               ]

for i, img_path in enumerate(next_apple_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

# plt.show()


# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255., rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2,shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

test_datagen = ImageDataGenerator( rescale = 1.0/255. )

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir, batch_size = 20, class_mode = 'binary', target_size = (224, 224))

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory( validation_dir,  batch_size = 20, class_mode = 'binary', target_size = (224, 224))

base_model = VGG16(input_shape = (224, 224, 3), # Shape of our images
include_top = False, # Leave out the last fully connected layer
weights = 'imagenet')

for layer in base_model.layers:
    layer.trainable = False


# Flatten the output layer to 1 dimension
x = layers.Flatten()(base_model.output)

# Add a fully connected layer with 512 hidden units and ReLU activation
x = layers.Dense(512, activation='relu')(x)

# Add a dropout rate of 0.5
x = layers.Dropout(0.5)(x)

# Add a final sigmoid layer with 1 node for classification output
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(base_model.input, x)
Optimizer = RMSprop(learning_rate=0.0001)

model.compile(optimizer = Optimizer, loss = 'binary_crossentropy',metrics = ['acc'])

model.fit(train_generator, validation_data = validation_generator, steps_per_epoch = 100, epochs = 10,  batch_size=32)

model.save('apple_classifier.h5')