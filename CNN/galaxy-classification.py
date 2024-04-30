# Import TensorFlow
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from numpy import *
import numpy as np


def model_setup():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    model.summary()

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    
    model.summary()
    # Adam is the best among the adaptive optimizers in most of the cases
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

# An epoch means training the neural network with all the
# training data for one cycle. Here I use 10 epochs
def model_fit(model, X_train, X_test, y_cat_train, y_cat_test):
    batch_size = 30
    data_generator = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    train_generator = data_generator.flow(X_train, y_cat_train, batch_size)
    steps_per_epoch = X_train.shape[0] // batch_size

    history = model.fit(train_generator, 
              epochs=20,
              steps_per_epoch=steps_per_epoch,
              validation_data=(X_test, y_cat_test), 
#               callbacks=[early_stop],
#               batch_size=batch_size,
             )

    # history = model.fit(train_images, train_labels, epochs=10, 
    #                     validation_data=(test_images, test_labels))
    return history

def show_accuracy():
    plt.figure(figsize=(7,5))
    plt.plot(history.history['accuracy'],label='accuracy')
    plt.plot(history.history['val_accuracy'],label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()
    
    return history

def show_predict(i):
    plt.figure(figsize=(5,5))
    plt.imshow(test_images[i])
    plt.title('actual='+class_names[ca[i]] + ', pred=' + class_names[cs[i]])
    plt.show()
    

from astroNN.datasets import load_galaxy10
from sklearn.model_selection import train_test_split

images, labels = load_galaxy10()

train_images, train_labels, test_images, test_labels = train_test_split(images, labels, test_size=0.1)

class_names = ['Disturbed', 'Merging', 'Round Smooth', 'In-between Round Smooth', 'Cigar-round Smooth', \
          'Barred Spiral', 'Unbarred Tight Spiral', 'Unbarred Loose Spiral', 'Edge-on Without Bulge', \
          'Edge-on with Bulge']

# Let's view more images in a grid format
# Define the dimensions of the plot grid 
W_grid = 5
L_grid = 2

# fig, axes = plt.subplots(L_grid, W_grid)
# subplot return the figure object and axes object
# we can use the axes object to plot specific figures at various locations

fig, axes = plt.subplots(L_grid, W_grid, figsize = (15,5))

axes = axes.ravel() # flaten the 15 x 15 matrix into 225 array

n_train = len(train_images) # get the length of the train dataset

# Select a random number from 0 to n_train
for i in np.arange(0, W_grid * L_grid): # create evenly spaces variables 

    # Select a random number
    index = np.random.randint(0, n_train)
    # read and display an image with the selected index    
    axes[i].imshow(train_images[index,1:])
    label_index = int(test_images[index])
    axes[i].set_title(class_names[label_index], fontsize = 8)
    axes[i].axis('off')

plt.subplots_adjust(hspace=0.4)
plt.show()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# model = model_setup()
# from tensorflow.keras.utils import to_categorical
# y_cat_train = to_categorical(test_images, 10)
# y_cat_test = to_categorical(test_labels, 10)
# history = model_fit(model, train_images, train_labels, y_cat_train, y_cat_test)

# yp = model.predict(test_images)
# cs = argmax(yp, axis=1)
# ca = test_labels.reshape(len(cs))

# show_accuracy()

# show_predict(0)
# show_predict(10)
# show_predict(20)
# show_predict(30)
              
