# Diabeties-detector

* An application for Diabetes detection. The app can tell whether the person has Diabetes or not and thus provide measures.
Partial implementation of MLP as described in this paper.

Dataset - pima-indians-diabetes

Multilayer perceptron neural network with 8 inputs in the visible layer, 32 and 16 neurons in the 2 hidden layer with ReLu activation function and 1 neuron in the output layer with sigmoid activation function.

Network trained for 700 epochs with batch size of 10 using ADAM optimizer and binary_crossentropy loss function.

Attribute Information:

Number of times pregnant
Plasma glucose concentration a 2 hours in an oral glucose tolerance test
Diastolic blood pressure (mm Hg)
Triceps skin fold thickness (mm)
2-Hour serum insulin (mu U/ml)
Body mass index (weight in kg/(height in m)^2)
Diabetes pedigree function
Age (years)
Class variable (0 or 1)*
Video : https://www.youtube.com/watch?v=fos8E_96waU

### Requirements ###

* Python >=2.7.14
* TensorFlow >= 1.4.0
* Flask
* Heroku
