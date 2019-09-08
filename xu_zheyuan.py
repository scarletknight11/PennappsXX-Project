#!/usr/bin/env python

##############
# Your name: Zheyuan Xu
##############

import numpy as np
import re
from sklearn import svm, metrics
from skimage import io, feature, filters, exposure, color

# additional packages for import
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
import cv2
from keras.preprocessing.image import img_to_array
import imutils
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

# helper class, structuring LeNet
class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # Initialize the model
        model = Sequential()
        input_shape = (height, width, depth)

        # If we are using 'channels-first', update the input shape
        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)

        # First set of CONV => RELU => POOL layers
        model.add(Conv2D(20, (5, 5), padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Second set of CONV => RELU => POOL layers
        model.add(Conv2D(50, (5, 5), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # First (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))

        # Softmax classifier
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        # return the constructed network architecture
        return model

# helping function, preprocessing the images
def preprocess(image, width, height):
    # Grab the dimensions of the image, then initialize the padding values
    (h, w) = image.shape[:2]

    # If the width is greater than the height then resize along the width
    if w > h:
        image = imutils.resize(image, width=width)
    # Otherwise, the height is greater than the width so resize along the height
    else:
        image = imutils.resize(image, height=height)

    # Determine the padding values for the width and height to obtain the target dimensions
    pad_w = int((width - image.shape[1]) / 2.0)
    pad_h = int((height - image.shape[0]) / 2.0)

    # Pad the image then apply one more resizing to handle any rounding issues
    image = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    # Return the pre-processed image
    return image

class ImageClassifier:

    def __init__(self):
        self.classifier = None

    def imread_convert(self, f):
        return io.imread(f).astype(np.uint8)

    def load_data_from_folder(self, dir):
        # read all images into an image collection
        ic = io.ImageCollection(dir+"*.bmp", load_func=self.imread_convert)

        # create one large array of image data
        data = io.concatenate_images(ic)

        # extract labels from image names
        labels = np.array(ic.files)
        for i, f in enumerate(labels):
            m = re.search("_", f)
            labels[i] = f[len(dir):m.start()]

        return(data, labels)

    def extract_image_features(self, data):
        # Please do not modify the header above

        # extract feature vector from image data

        ########################
        # YOUR CODE HERE
        # load the image, preprocess it, and store it in the data list
        featured_data = []
        for image in data:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # scale into 28X28 arrays
            image = preprocess(image, 28, 28)
            image = img_to_array(image)
            featured_data.append(image)
        featured_data = np.array(featured_data, dtype="float") / 255.0
        ########################
        # Please do not modify the return type below
        return(featured_data)

    def train_classifier(self, train_data, train_labels):
        # Please do not modify the header above

        # train model and save the trained model to self.classifier

        ########################
        # YOUR CODE HERE
        # split the training dataset into training and validation set
        (train_x, test_x, train_y, test_y) = train_test_split(train_data, train_labels, test_size=0.05, random_state=42)
        # Convert the labels from integers to vectors
        lb = LabelBinarizer().fit(train_y)
        train_y = lb.transform(train_y)
        test_y = lb.transform(test_y)
        # Initialize the model
        model = LeNet.build(width=28, height=28, depth=1, classes=8)
        optimizer = SGD(lr=0.01, decay=1e-8, momentum=0.9, nesterov=True)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

        # Train the network
        print("[INFO]: Training....")
        H = model.fit(np.array(train_x), np.array(train_y), validation_data=(np.array(test_x), np.array(test_y)), batch_size=16, epochs=70, verbose=1)

        # parse the result to classifier
        self.classifier = model
        ########################

    def predict_labels(self, data):
        # Please do not modify the header

        # predict labels of test data using trained model in self.classifier
        # the code below expects output to be stored in predicted_labels

        ########################
        # YOUR CODE HERE
        predicted_labels = []
        label_list = ['drone', 'hands', 'inspection', 'none', 'order', 'place', 'plane', 'truck']
        for pic in data:
            # note: -1 is added since it expects input dimension of 4
            pic = pic.reshape(-1,28, 28, 1)
            result = self.classifier.predict(pic)
            result = result[0]
            result = result.tolist()
            idx = result.index(max(result))
            print(idx)
            label = label_list[idx]
            predicted_labels.append(label)
            print(label)
        ########################
        # Please do not modify the return type below
        return predicted_labels

def main():

    img_clf = ImageClassifier()

    # load images
    (train_raw, train_labels) = img_clf.load_data_from_folder('./train/')
    (test_raw, test_labels) = img_clf.load_data_from_folder('./test/')

    # convert images into features
    train_data = img_clf.extract_image_features(train_raw)
    test_data = img_clf.extract_image_features(test_raw)

    # train model and test on training data
    img_clf.train_classifier(train_data, train_labels)
    predicted_labels = img_clf.predict_labels(train_data)
    print("\nTraining results")
    print("=============================")
    print("Confusion Matrix:\n", metrics.confusion_matrix(
        train_labels, predicted_labels))
    print("Accuracy: ", metrics.accuracy_score(train_labels, predicted_labels))
    print("F1 score: ", metrics.f1_score(
        train_labels, predicted_labels, average='micro'))

    # test model
    predicted_labels = img_clf.predict_labels(test_data)
    print("\nTest results")
    print("=============================")
    print("Confusion Matrix:\n", metrics.confusion_matrix(
        test_labels, predicted_labels))
    print("Accuracy: ", metrics.accuracy_score(test_labels, predicted_labels))
    print("F1 score: ", metrics.f1_score(
        test_labels, predicted_labels, average='micro'))


if __name__ == "__main__":
    main()
