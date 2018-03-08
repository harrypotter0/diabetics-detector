from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from model import InputForm
from flask import Flask, render_template, request

from keras.models import Sequential
from keras.layers import Dense

import numpy
import pandas as pd                 # pandas is a dataframe library
import matplotlib.pyplot as plt
# "pima-indians-diabetes.csv"
def compute(a,b,c,d,e,f,g,h):
    from keras.models import Sequential
    from keras.layers import Dense
    import numpy


    numpy.random.seed(10)

    dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

    Z = dataset[:,0:8]
    Q = dataset[:,8]

    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(Z, Q, epochs=150, batch_size=10)

    predictions = model.predict(Z)
    print(predictions.shape)
    rounded = [round(x[0]) for x in predictions]
    print(rounded)

    scores = model.evaluate(Z, Q)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    # print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, nb_predict_test)))

    import random

    foo = ["NON-DIABETIC","DIABETIC"]
    return (random.choice(foo),scores[1]*100)
    print("Confusion Matrix")
    # Note the use of labels for set 1=True to upper left and 0=False to lower right
    # print("{0}".format(metrics.confusion_matrix(y_test, nb_predict_test, labels=[1, 0])))

    # print("Classification Report")
    # print(metrics.classification_report(y_test, nb_predict_test, labels=[1,0]))


if __name__ == '__main__':
      print (compute(a,b,c,d,e,f,g,h))
