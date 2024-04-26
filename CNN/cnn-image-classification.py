# Import TensorFlow
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from numpy import *


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
def model_fit(model):
    history = model.fit(train_images, train_labels, epochs=10, 
                        validation_data=(test_images, test_labels))
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
    

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(8,8))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays,
    #which is why we need the extra index
plt.xlabel(class_names[train_labels[i][0]])
plt.show()

model = model_setup()
history = model_fit(model)

yp = model.predict(test_images)
cs = argmax(yp, axis=1)
ca = test_labels.reshape(len(cs))

show_accuracy()

show_predict(0)
show_predict(10)
show_predict(20)
show_predict(30)
              
