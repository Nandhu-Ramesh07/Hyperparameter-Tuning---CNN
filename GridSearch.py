import tensorflow as tf
import cv2
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
import keras_tuner as kt
from kerastuner.tuners import GridSearch
from tensorflow import keras

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

def model_builder(hp):
  model = keras.Sequential()
  model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))),
  model.add(keras.layers.MaxPooling2D((2, 2))),
  model.add(keras.layers.Conv2D(64, (3, 3), activation='relu')),
  model.add(keras.layers.MaxPooling2D((2, 2))),
  model.add(keras.layers.Flatten(input_shape=(28, 28)))
  # Tune the number of units in the first Dense layer
  # Choose an optimal value between 32-512
  hp_units = hp.Int('units', min_value=32, max_value=512, step=64)
  model.add(keras.layers.Dense(units=hp_units, activation='relu'))
  model.add(keras.layers.Dense(10))

  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  return model

tuner = kt.GridSearch(model_builder,
                      objective='val_accuracy',
                      max_trials=5,  # Number of trials (combinations) to evaluate
                      directory=r"C:\Users\nandh\Desktop\Nandhu Ramesh_DA with Computational Science_Set5\Grid Search\my_dir",
                      project_name='GridSearchFiles')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(train_images, train_labels, epochs=5, validation_split=0.2,batch_size =128, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

# Build the model with the optimal hyperparameters and train it on the data for 5 epochs
model = tuner.hypermodel.build(best_hps)
history = model.fit(train_images, train_labels, epochs=5,batch_size =128, validation_split=0.2)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
import time
t1=time.time()
hypermodel.fit(train_images, train_labels, epochs=best_epoch,batch_size =128, validation_split=0.2)
print("Run Time = ",time.time()-t1)
eval_result = hypermodel.evaluate(test_images, test_labels)
print("[test loss, test accuracy]:", eval_result)