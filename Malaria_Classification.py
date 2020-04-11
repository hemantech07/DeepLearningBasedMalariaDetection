import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing import image
import pickle

train_dir = 'cell_images\Train'
validation_dir = 'cell_images\Validation'

train_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    target_size=(100,100),
                                                    class_mode='binary')

validation_datagen = ImageDataGenerator(rescale = 1./255 )

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              batch_size=10,
                                                              target_size=(100,100),
                                                              class_mode='binary')

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.95):
            print("\n Desired Accuracy Reached")
            self.model.stop_training = True

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(128, (3,3), activation='relu', input_shape=(100,100,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

callbacks = myCallback()

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])

history = model.fit_generator(train_generator,
                              epochs=15,
                              validation_data=validation_generator,
                              validation_steps=10,
                              callbacks = [callbacks],
                              verbose=1)

model.save('Malaria.h5')