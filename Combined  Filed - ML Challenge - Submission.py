#!/usr/bin/env python
# coding: utf-8

# # Models

# In[1]:


## Imports 

import numpy as np
from sklearn.model_selection import train_test_split as tts
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelBinarizer
import string
from keras.optimizers import Adam, SGD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LogisticRegression
import math
import cv2
from os.path import join
from os import getcwd


##

## Task 1: training two models

## Model 1: Multilayer perceptron 

## import data set 
path = join( getcwd(), 'data', 'training-dataset.npz' ) 

with np.load(path) as data: 
    
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
model.add(Dropout(0.4))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(unique_labels, activation='softmax'))

callback = EarlyStopping(monitor='val_loss', patience=4)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

run_model = model.fit(X_train, y_train_hot, batch_size=256, epochs=25, verbose=1,
                   validation_data=(X_val, y_val_hot), callbacks=[callback])

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
model2.add(Dense(256, input_dim=(x_train.shape[1]), activation='relu'))
model2.add(Dropout(0.4))

model2.add(Dense(256, activation='relu'))
model2.add(Dropout(0.4))

model2.add(Dense(256, activation='relu'))
model2.add(Dropout(0.4))

model2.add(Dense(Y_train.shape[1], activation='softmax')) #last layer

es_callback = EarlyStopping(monitor='val_loss', patience=5)

optimizer = Adam(lr=0.001)
model2.compile(optimizer=optimizer, loss = "categorical_crossentropy", metrics=['accuracy']) 
model2.fit(x_train, Y_train, batch_size=32, epochs=50, verbose=1, validation_data=(x_val, Y_val), callbacks=[es_callback])

y_pred = model2.predict(x_val)
print('MAE:', mean_absolute_error(Y_val, y_pred).round(3))
print('MSE:', mean_squared_error(Y_val, y_pred).round(3))

baseline = LogisticRegression()
baseline.fit(x_val, y_val)

print('Accuracy Validation:', accuracy_score(y_val, baseline.predict(x_val)).round(3))

[test_loss, test_acc] = model2.evaluate(x_test, Y_test)
print("Results: Loss = {}, accuracy = {}".format(test_loss, test_acc))


# # Task 2

# ## Additional Model

# In[6]:


from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout

from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split



#Prep data 
##Normalisation
img = img/ 255.0 #float value to keep values consitent. 

##Reshape 
img = img.reshape(-1,28,28,1) #Necesssary for CNN (more info Rosenbrock, 2017, p181)

#Train (0.8) / Test (0.2)
X_train, X_test, y_train, y_test= train_test_split(img, lbl, test_size=0.2, random_state=1)

#Test (0.60)/ val (0.25*0.8 = 0.2)
X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

#onehot encoding
onehot = LabelBinarizer()
y_train_hot = onehot.fit_transform(y_train)
y_val_hot   = onehot.transform(y_val)
y_test_hot  = onehot.transform(y_test)

# Set a learning rate annealer (Ghouzam, 2017)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

epochs = 1 
batch_size = 80

# data augmentation to prevent overfitting (Ghouzam, 2017)
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=3,  # randomly rotate images in the range (degrees, 0 to 54) mot 90 as ME
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images - why bd, mw
        vertical_flip=False)  # randomly flip images - pd

datagen.fit(X_train)

#CNN model (adapted from LeCun, 1998)
model3 = Sequential()

# first set of CONV => RELU => POOL layers
model3.add(Conv2D(filters=20,
                  kernel_size = (5, 5), # filter matrix
                  padding = 'same', # preservs borders 
                  activation = 'relu', #activation function 
                  input_shape = (28, 28, 1))) #necessary input format
model3.add(Conv2D(filters=20,
                  kernel_size = (5, 5), # filter matrix
                  padding = 'same', # preservs borders 
                  activation = 'relu'))
model3.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))) #downsampling
model3.add(Dropout(0.0625)) #prevent overfitting 

# second set of CONV => RELU => POOL layers
model3.add(Conv2D(filters=50,
                  kernel_size = (5, 5), # filter matrix
                  padding = 'same', # preservs borders 
                  activation = 'relu')) #necessary input format
model3.add(Conv2D(filters=50,
                  kernel_size = (5, 5), # filter matrix
                  padding = 'same', # preservs borders 
                  activation = 'relu')) #necessary input format
model3.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))) #downsampling
model3.add(Dropout(0.0625))

# first (and only) set of FC => RELU layers
model3.add(Flatten())
model3.add(Dense(500, activation = 'relu'))
model3.add(Dropout(0.125))

# softmax classifier
model3.add(Dense(26, activation = "softmax"))

# Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# Compile the model
model3.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

#fit 
history = model3.fit_generator(datagen.flow(X_train,y_train_hot, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,y_val_hot),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])


# ## Create list of models

# In[7]:


## Create list of models
list_of_model = [model,model2, model3]


# ## Supporting functions

# In[8]:


## Define upporting functions 

#### resizing image while mainiting aspect ration 
def image_resize(image, width, height, inter = cv2.INTER_AREA):
    
    # grab the image size
    (h, w) = image.shape[:2]

    if w > h: 
        r = width / float(w)
        dim = (width, int(h * r))
        resized = cv2.resize(image, dim, interpolation = inter)
        
    else: 
        r = height / float(h)
        dim = (int(w * r), height)
        resized = cv2.resize(image, dim, interpolation = inter)
 
    # define pad size
    hpad = (height - resized.shape[0]) /2
    wpad = (width - resized.shape[1]) /2 
    
    tpad = int(math.ceil(hpad)) # round up
    bpad = int(math.floor(hpad)) # round down
    lpad = int(math.ceil(wpad)) # round up
    rpad = int(math.floor(wpad)) #round down
    
    # pad the image then apply one more resizing to handle any
    # rounding issues
    image = cv2.copyMakeBorder(resized, tpad, bpad, lpad, rpad,cv2.BORDER_CONSTANT)
    
    return image
    
### finds appropiate contours
def find_contour(dic): #dictionary with rectangle contour & index as key
    
    #Step 2: sort dictionary with entry with longest width
        ###NOTE: Assumption - Overlapping letters are of longest width 
    longest_width = sorted(dic.values(), key=lambda x:x[2])[-1] #longest one last 

    #Step 3: Create new contours 
    new_contours = {}
    counter = 0 

    for ind in range( len(dic) ): 

        #separate letters 
        if dic[ind] == longest_width: 

            #setup
            x, y, w, h = dic[ind] 

            #first letter 
            r1 = x, y, int(w/2), h 
            new_contours[counter] = r1
            counter +=1 

            #second letter 
            r2 = int(x + w/2), y, int(w/2), h
            new_contours[counter] = r2     
            counter +=1 
            

        else: 
            new_contours[counter]= dic[ind]
            counter +=1   
                   
    return new_contours


### removes contours of too small size
def unwanted(dic): #List of array of contours// assumes contour more than 4 
        
    # list of array by how many too long 
    no_excess = len(dic) - 4
    
    #list showing array and size of respective array
    l = []
    for contour in dic.values():
        
        #setup
        x, y, w, h = contour

        l.append( (contour, (w*h)) )
        

    #create new list with excess array
    excess =  sorted(l, key=lambda x:x[-1])[:no_excess]
    re_list = []
    
    for item in excess:
        
        re_list.append(item[0])

    #Step 3: Create new contours 
    new_contours = {}
    counter = 0 

    for val in dic.values():
        
        if val in re_list: 
            
            continue 
            
        else: 
            new_contours[counter] = val
            counter +=1
            
    return new_contours

### find 4 contour if we do not have 4 contours 
def find_4contours(contours_found): # dic with index keys 
    
    if len( contours_found ) < 4:
            
            new_contours = find_contour(contours_found) # here recursion would be good 
            
    else:
            
            new_contours = unwanted(contours_found) #removes excess array based on size
            
    return new_contours
            

### check if we have 4 contours in recursive fashion 
def check(contours_found):
    
    if len(contours_found) == 4: 
        
        return contours_found
    
    else: 
        return check(find_4contours(contours_found))


# ## Execution

# In[9]:


## Execution of Part 2

#load dataset 
path = join( getcwd(), 'data', 'test-dataset.npy' ) 
captchas = np.load(path).astype('uint8') # test_ll = test_labelless


#iterate through case by case: 
final_pred = []

for captcha in captchas: 
    
    #------format captcha image-----#
    
    ##add extra padding around the image
    pad = cv2.copyMakeBorder(captcha, 8, 8, 8, 8, cv2.BORDER_CONSTANT)
    
    ## blur 
    blur = cv2.medianBlur(pad , 3)
    
    ##thresholding 
    _, thresh = cv2.threshold(blur,0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) #127 middle between white/black
    
    
    #------contour extraction-----#
   
    ##detect contours of all images 
    contours, _ = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    #print (contours, f"hierachy: {hierachy}")
    #------contour filtration-----#
    
    ##contours smaller than 30 pixels excluded
            ##NOTE - replacing pixel with black may lead to imporvement
    
    contours_letters = []
    
    for contour in contours:

        if contour.size > 30: # check whats the best size 
            
            contours_letters.append(contour)
            
            
    #------letter extraction-----#  
    
        #Prep: Build dic with contours and index as key
        
    contours_extracted = [] #stores contours found: rectangle
                        #e.g. [(103, 9, 20, 17), ...]
    
    for i in range( len(contours_letters) ):
        
        contours_extracted.append(cv2.boundingRect(contours_letters[i]))
        
        
    #------sort contours-----# 
     ## sorted in accordance with x position
    sort_x = sorted(contours_extracted, key=lambda x: x[0])
    
    ##index key position
    contours_found = {}
    
    for ind, contour in enumerate( sort_x ):
        
        contours_found[ind] = contour
        
    #--contour rectangle adjust--#
    
        #Main part: Finding relevant 4 contours if 
        #we do not have more or less than 4 contours
    
    new_contours = check( contours_found ) #this works 
        
    #-------prediction-------
    
        #take dictionary with index as key in form {0: (103, 9, 20, 17), ...}
        
    pred5 =[] #stores 5 pred
    for model in list_of_model:
        
        captcha = ""
        for key in new_contours.keys():

            x, y, w, h = new_contours[key]

            # Extract the letter from the original image with a 2-pixel margin around the edge
            #letter_image = pad[y - 2:y + h + 2, x - 2:x + w + 2]
            letter_image = pad[y:y + h, x:x + w]

            #resize
            resized = image_resize(letter_image, 28, 28)
            
            #Normalisation/ reshape
            resized_std = resized/255.0  # Normalisation 

            ##conditional reshape
            if model == model3:
                input_for_model = resized_std.reshape(-1,28,28,1)
            else:
                input_for_model = resized_std.reshape(-1,784)
                
            #prediction
            pred = model.predict(input_for_model)

            #append list 
            letter = onehot.inverse_transform(pred)[0]

            #correct format
            if letter < 10:
                captcha += "0" + str( letter )
            else:
                captcha += str( letter ) 
        
        
        pred5.append(captcha)
        
    final_pred.append(pred5)
    print(pred5)
    
    
## Store pred in csv 
csv_pred = np.array(final_pred)
np.savetxt("predictions.csv", csv_pred, fmt="%s", delimiter=",")

