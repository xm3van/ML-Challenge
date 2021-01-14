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
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools

##

## Task 1: training two models

## Model 1: Multilayer perceptron 

## import data set 
with np.load('training-dataset.npz') as data: 
    
    img = data['x']
    lbl = data['y']
    
img = img / 255.0
    
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

## ImageDataGenerator function for data augmentation

datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False, 
        featurewise_std_normalization=False, 
        samplewise_std_normalization=False, 
        zca_whitening=False, 
        rotation_range=3,  
        zoom_range = 0.1, 
        width_shift_range=0.1, 
        height_shift_range=0.1, 
        horizontal_flip=False, 
        vertical_flip=False)


datagen.fit(X_train.reshape(-1, 28, 28, 1))

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

print("Results multi level perceptron: Loss = {}, accuracy = {}".format(test_loss, test_acc))

## Training and validation loss, training and validation accuracy

fig, ax = plt.subplots(2,1)
ax[0].plot(run_model.history['loss'], color='b', label="Training loss")
ax[0].plot(run_model.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(run_model.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(run_model.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

## 

## Model 2: Neural Network

x, x_test, y, y_test = tts(img, lbl, test_size=0.15, train_size=0.85) 
x_train, x_val, y_train, y_val = tts(x, y, test_size=0.17647059, train_size=0.82352941) 

LB = LabelBinarizer()
Y_train = LB.fit_transform(y_train)
Y_val   = LB.transform(y_val)
Y_test  = LB.transform(y_test)

datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False, 
        featurewise_std_normalization=False, 
        samplewise_std_normalization=False, 
        zca_whitening=False, 
        rotation_range=3,  
        zoom_range = 0.1, 
        width_shift_range=0.1, 
        height_shift_range=0.1, 
        horizontal_flip=False, 
        vertical_flip=False)

datagen.fit(x_train.reshape(-1, 28, 28, 1))

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

## Confusion matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Predict the values from the validation dataset
Y_pred = model2.predict(x_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(26)) 

## Errors

errors = (Y_pred_classes - Y_true != 0)

Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = Y_pred[errors]
Y_true_errors = Y_true[errors]
X_val_errors = x_val[errors]

def display_errors(errors_index,img_errors,pred_errors, obs_errors):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((28,28)))
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
            n += 1

# Probabilities of the wrong predicted numbers
Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)

# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))

# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

# Sorted list of the delta prob errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 6 errors 
most_important_errors = sorted_dela_errors[-6:]

# Show the top 6 errors
display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)

#############################
# # Task 2
#############################

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


# In[167]:


#Extract most likely set of 2 from prediction of one letter
def top_2_letter(pred_array):
    #index dic 
    pred_prob = {}
    for ind, prob in enumerate(pred_array[0]):
        pred_prob[ind+1] = prob

    #sort
    highest_prob_t5 = sorted(pred_prob.items(), key=lambda x:x[-1])[-2:]

    return highest_prob_t5


# In[175]:


#extract top 5 most likely prediction of set with 4 letter prediction
#with respective 2 set of prediction and prob
def extract_top_5(test):

    #Combination set of zero & ones
    combi_set =[]
    for l1 in range(2):
        for l2 in range(2):
            for l3 in range(2):
                for l4 in range(2):
                    combi_set.append([l1,l2,l3,l4])
 
    #extract cobination from most set
    p_set ={}   
    for combi in combi_set:

            cap = "" 
            prob_sum = 0

            for ind, c in enumerate(combi): 
                
                #if number smaller than 10                
                if test[ind][c][0] < 10:
                    cap += "0" + str(test[ind][c][0])
                else:
                    cap += str(test[ind][c][0]) #index
                
                prob_sum += test[ind][c][1] #prob 

            p_set[int(cap)] = prob_sum

    #extrat 5 most likely predictions
    t5_list = sorted(p_set.items(), key=lambda x:x[-1], reverse = True)[:5]

    #print(cap, prob_sum)
    p5 = []
    for i in t5_list: 
        p5.append( i[0])

    return p5



# In[176]:

###################################
## Execution of Part 2
###################################

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
    
    set_top2_prediction = []
    for key in new_contours.keys():

            x, y, w, h = new_contours[key]

            # Extract the letter from the original image with a 2-pixel margin around the edge
            #letter_image = pad[y - 2:y + h + 2, x - 2:x + w + 2]
            letter_image = pad[y:y + h, x:x + w]

            #resize
            resized = image_resize(letter_image, 28, 28)
            
            #Normalisation/ reshape
            resized_std = resized/255.0  # Normalisation 
            
            #reshape for model
            input_for_model = resized_std.reshape(-1,784)
                
            #prediction matrix with prob
            pred = model.predict(input_for_model)
            
            #top 2 pred per letter       
            set_top2_prediction.append( top_2_letter(pred) )
            
    pred5.append(set_top2_prediction)
    
    top_5 = extract_top_5(pred5[0])
                
    final_pred.append(top_5)
   
    
    
## Store pred in csv 
csv_pred = np.array(final_pred)
np.savetxt("prediction.csv", csv_pred, fmt='%08d', delimiter=",")

