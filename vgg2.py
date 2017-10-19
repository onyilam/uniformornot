import numpy as np
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, BatchNormalization
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os, math
from keras.models import Model

# path to the model weights files.
weights_path = 'vgg16_weights.h5'
# dimensions of our images.
img_width, img_height = 150, 150
top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'train'
validation_data_dir = 'validation'

nb_train_uniform = len(os.walk('train/uniform').next()[2])-1 #dir is your directory path as string
nb_train_notuniform = len(os.walk('train/notuniform').next()[2])-1 #dir is your directory path as string
nb_train_samples = nb_train_uniform + nb_train_notuniform
print 'number of training sample:', nb_train_uniform, nb_train_notuniform, nb_train_samples
nb_validation_uniform = len(os.walk('validation/uniform').next()[2]) -1  #dir is your directory path as string
nb_validation_notuniform = len(os.walk('validation/notuniform').next()[2]) -1 #dir is your directory path as string
nb_validation_samples = nb_validation_notuniform  + nb_validation_uniform

print 'number of validation sample:', nb_validation_notuniform, nb_validation_uniform, nb_validation_samples

epochs = 50
batch_size = 16

# build the VGG16 network
base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
print('Model loaded.')


# build a classifier model to put on top of the convolutional model
top_model = Sequential()
#flatten since too many dimensions, we only want a classification output

top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
#top_model.add(BatchNormalization())
top_model.add(Dense(256, activation='relu'))
#randomly turn neurons on and off to improve convergence
top_model.add(Dropout(0.5))
#fully connected to get all relevant data
top_model.add(Dense(1, activation='sigmoid'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)


# add the model on top of the convolutional base
model = Model(input= base_model.input, output= top_model(base_model.output))
#model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:15]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    batch_size=batch_size,
    class_mode='binary')

model.summary()

# fine-tune the model
model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples/batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples/batch_size)

model.save_weights('finetune_vgg_model.h5')
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
