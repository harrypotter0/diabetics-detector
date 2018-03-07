from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pandas as pd
from model import InputForm
from flask import Flask, render_template, request

def compute(a,b,c,d,e,z,g,h):
    data_file_name = 'pima_indians_diabetes.txt'
    first_line = "pregnant, glucose, bp, skin, insulin, bmi, pedigree, age, label"
    with open(data_file_name, "r+") as f:
         content = f.read()
         f.seek(0, 0)
         f.write(first_line.rstrip('\r\n') + '\n' + content)
    df = pd.read_csv(data_file_name)
    df.replace('?', np.nan, inplace = True)
    df.dropna(inplace=True)
    # df.drop(['id'], axis = 1, inplace = True)

    # df['class'].replace('1',0, inplace = True)
    # df['class'].replace('0',1, inplace = True)

    df.to_csv("combined_data.csv", index = False)

    # Data sets
    DIABETES_TRAINING = "diabetes_train.csv"
    DIABETES_TEST = "diabetes_test.csv"

    ######################################
    # classifier
    # Load datasets
    training_set_classifier = tf.contrib.learn.datasets.base.load_csv_with_header(filename=DIABETES_TRAINING,
                                                     target_dtype=np.float32, features_dtype=np.float32)
    test_set_classifier =  tf.contrib.learn.datasets.base.load_csv_with_header(filename=DIABETES_TEST,
                                                 target_dtype=np.float32, features_dtype=np.float32)

    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=8)]

    classifier_classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[8, 16, 8],
                                            n_classes=2,
                                            model_dir="/tmp/iris_model")
    #####################################
    # Accuracy
    # Load datasets.
    training_set_accuracy = tf.contrib.learn.datasets.base.load_csv_with_header(filename=DIABETES_TRAINING,
                                                           target_dtype=np.float32,
                                                           features_dtype=np.float32,
                                                           target_column=-1)
    test_set_accuracy = tf.contrib.learn.datasets.base.load_csv_with_header(filename=DIABETES_TEST,
                                                       target_dtype=np.float32,
                                                       features_dtype=np.float32,
                                                       target_column=-1)

    # Specify that all features have real-value data
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=2)]

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier_accuracy = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[8, 16, 8],
                                                n_classes=2,
                                                model_dir="/tmp/iris_model")

    ##########################################

    #classifier
    # Fitting model
    classifier_classifier = classifier_classifier.fit(training_set_classifier.data,
                            training_set_classifier.target,
                            steps=2000)
    k =a
    l = b

    m =c
    n= d

    o = e
    p = z

    q = g
    r = h
    # s  = i
    '''
10,115,0,0,0,35.3,0.134,29,0
    '''

    def new_samples():
        return np.array([[k, l, m, n, o, p, q, r],
                 ], dtype=np.float32)

    s = list(classifier_classifier.predict(input_fn=new_samples))

    # Accuracy
    # Fit model.
    classifier_accuracy.fit(x=training_set_accuracy.data,
                   y=training_set_accuracy.target,
                   steps=2000)

    # Evaluate accuracy.
    accuracy_score = classifier_accuracy.evaluate(x=test_set_accuracy.data,
                                         y=test_set_accuracy.target)["accuracy"]

    if (s == [[1]]):
          return ("DIABETIC",accuracy_score*100)
    else:
          return ("NOT DIABETIC",accuracy_score*100)


if __name__ == '__main__':
      print (compute(a,b,c,d,e,f,g,h))
