## Imports 

import numpy as np
from sklearn.model_selection import train_test_split as tts
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelBinarizer
import string
from keras.optimizers import Adam, SGD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LogisticRegression

##

## Task 1: training two models

## Model 1: Multilayer perceptron 

with np.load('training-dataset.npz') as data:
    img = data['x']
    lbl = data['y']
    
X_train, X_test, y_train, y_test = tts(img, lbl, test_size=0.2)
X_train, X_val, y_train, y_val = tts(X_train, y_train, test_size=0.25)

print("Training data shape:", X_train.shape, y_train.shape)
print("Testing data shape:", X_test.shape, y_test.shape)

unique_labels = len(set(lbl))
labels = set(lbl)
print("Number of labels:", unique_labels)
print("Output classes:", labels)

onehot = LabelBinarizer()
y_train_hot = onehot.fit_transform(y_train)
y_val_hot = onehot.transform(y_val)
y_test_hot = onehot.transform(y_test)

inp_shape = X_train.shape[1]

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(inp_shape,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(unique_labels, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
run_model = model.fit(X_train, y_train_hot, batch_size=256, epochs=50, verbose=1,
                   validation_data=(X_val, y_val_hot))

[test_loss, test_acc] = model.evaluate(X_test, y_test_hot)

print("Results model with 4 layers: Loss = {}, accuracy = {}".format(test_loss, test_acc))


##

## Model 2: Neural Network

x, x_test, y, y_test = tts(img, lbl, test_size=0.15, train_size=0.85) 
x_train, x_val, y_train, y_val = tts(x, y, test_size=0.17647059, train_size=0.82352941) 

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

LB = LabelBinarizer()
Y_train = LB.fit_transform(y_train)
Y_val   = LB.transform(y_val)
Y_test  = LB.transform(y_test)

model2 = Sequential()
model2.add(Dense(512, input_dim=784, activation='relu'))
model2.add(Dense(512, activation='relu'))
model2.add(Dense(512, activation='relu'))
model2.add(Dense(26, activation='softmax')) #last layer

optimizer = Adam(lr=0.001)
model2.compile(optimizer=optimizer, loss = "categorical_crossentropy", metrics=['accuracy']) 
model2.fit(x_train, Y_train, batch_size=32, epochs=50, verbose=1)

y_pred = model2.predict(x_val)
print('MAE:', mean_absolute_error(Y_val, y_pred).round(3))
print('MSE:', mean_squared_error(Y_val, y_pred).round(3))

baseline = LogisticRegression()
baseline.fit(x_val, y_val)

print('Accuracy Validation:', accuracy_score(y_val, baseline.predict(x_val)).round(3))

[test_loss, test_acc] = model2.evaluate(x_test, Y_test)
print("Results: Loss = {}, accuracy = {}".format(test_loss, test_acc))

##

## Task two: Running on multiple letters within image

##

## Extra: Predict function to use on trained models

def predict_letter(testcase, model, xtest, ytest): # input is a number between 0 and 24960
    model_name = model
    alphabet = list(string.ascii_lowercase) # list of alphabet letters for clarity
    prediction = alphabet[model_name.predict_classes(xtest[[testcase], :])[0]] # use the model to predict a one hot label, convert to letter
    true_label = alphabet[ytest[testcase] -1] # true label from y_test one hot, converted to a letter
    
    print("Letter prediction {}".format(prediction))
    plt.imshow(X_test[testcase].reshape((28, 28)), cmap='gray') # plot the letter for clarity
    plt.title("True label: {}".format(true_label))
    plt.show()
    print("Actual letter: {}".format(true_label))
    if true_label == prediction:
        print("The model got it correct")
    else:
        print("The model got it wrong")
    
    return
    
#predict_letter(863, model, X_test, y_test)

##