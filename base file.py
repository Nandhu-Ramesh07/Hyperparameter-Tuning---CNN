import tensorflow as tf
import cv2
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import classification_report

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')  # 10 classes for Fashion-MNIST
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
strt=time.time()
history=model.fit(train_images, train_labels, epochs=5, batch_size =128,validation_split=0.1)  # Adjust epochs as needed
end=time.time()
print("Run Time = ",end-strt)


# Plot and save accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.savefig(r'C:\Users\nandh\Desktop\Nandhu Ramesh_DA with Computational Science_Set5\Base File using CNN\results_mnist\mnist_accuracy_plot.png')

plt.clf()


# Plot and save loss
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig(r'C:\Users\nandh\Desktop\Nandhu Ramesh_DA with Computational Science_Set5\Base File using CNN\results_mnist\mnist_loss_plot.png')

print("--------------------------------------\n")
print("Model Evalutaion Phase.\n")
loss,accuracy=model.evaluate(test_images, test_labels)
print(f'Accuracy: {round(accuracy*100,2)}')
print("--------------------------------------\n")

def make_prediction(img, model):
    # Read the image (assuming a grayscale image)
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)  # Load as grayscale

    # Resize to Fashion-MNIST dimensions (28x28)
    img = cv2.resize(img, (28, 28))

    # Normalize pixel values to the range [0, 1]
    img = img / 255.0

    # Add a channel dimension (for CNN compatibility)
    input_img = np.expand_dims(img, axis=0)

    # Make a prediction using the model
    res = model.predict(input_img)

    # Interpret the prediction based on Fashion-MNIST classes
    predicted_class = np.argmax(res)
    predicted_label = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"][predicted_class]

    print("Predicted label:", predicted_label)

make_prediction(r"C:\Users\nandh\OneDrive\Pictures\Screenshots\Screenshot 2023-12-30 080635.png",model)
make_prediction(r"C:\Users\nandh\OneDrive\Pictures\Screenshots\Screenshot 2023-12-30 080648.png",model)
make_prediction(r"C:\Users\nandh\OneDrive\Pictures\Screenshots\Screenshot 2023-12-30 080657.png",model)
make_prediction(r"C:\Users\nandh\OneDrive\Pictures\Screenshots\Screenshot 2023-12-30 080705.png",model)
make_prediction(r"C:\Users\nandh\OneDrive\Pictures\Screenshots\Screenshot 2023-12-30 080714.png",model)
make_prediction(r"C:\Users\nandh\OneDrive\Pictures\Screenshots\Screenshot 2023-12-30 080722.png",model)
make_prediction(r"C:\Users\nandh\OneDrive\Pictures\Screenshots\Screenshot 2023-12-30 080732.png",model)
make_prediction(r"C:\Users\nandh\OneDrive\Pictures\Screenshots\Screenshot 2023-12-30 080739.png",model)