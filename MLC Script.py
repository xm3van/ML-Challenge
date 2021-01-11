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
import math
import cv2
from os.path import join
from os import getcwd

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
run_model = model.fit(X_train, y_train_hot, batch_size=256, epochs=3, verbose=1,
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
model2.fit(x_train, Y_train, batch_size=32, epochs=3, verbose=1)

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

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    #return resized
    # define padding 
    if resized.shape[0] < height: 
        pad = (height - resized.shape[0]) /2
        tpad = math.ceil(pad) # round up
        bpad = int(pad) # round down
        lpad, rpad = 0,0 
        
    if resized.shape[1] < width:
        pad = (width - resized.shape[1])/2
        lpad = math.ceil(pad) # round up
        rpad = int(pad)
        tpad, bpad = 0, 0 
        
    image = cv2.copyMakeBorder(resized, tpad, bpad, lpad, rpad, # top, bottom, left, right,
        cv2.BORDER_CONSTANT)
   
    return image

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
                   
    return new_contours # contours_found (for testing > shows change)

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

def find_4contours(contours_found): # dic with index keys 
    
    if len( contours_found ) < 4:
            
            new_contours = find_contour(contours_found) # here recursion would be good 
            
    else:
            
            new_contours = unwanted(contours_found) #removes excess array based on size
            
    return new_contours

def check(contours_found):
    
    if len(contours_found) == 4: 
        
        return contours_found
    
    else: 
        return check(find_4contours(contours_found))
    

#path = join( getcwd(), 'data', 'test-dataset.npy' ) 
captchas = np.load("test-dataset.npy")

#iterate through case by case: 
final_pred = []
css = []

for exp, captcha in enumerate (captchas): 
    
    #------format captcha image-----#
    
    ##add extra padding around the image
    pad = cv2.copyMakeBorder(captcha, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
    
    ## blur 
    blur = cv2.medianBlur(pad , 3)
    
    ##thresholding 
    _, thresh = cv2.threshold(blur,127,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) #127 middle between white/black
    
    
    #------contour extraction-----#
   
    ##detect contours of all images 
    contours, hierachy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    #print (contours, f"hierachy: {hierachy}")
    #------contour filtration-----#
    
    ##contours smaller than 30 pixels excluded
            ##NOTE - replacing pixel with black may lead to imporvement
    
    contours_letters = []
    
    for contour in contours:
        
        css.append(contour.size)

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
    
    captcha = ""
    
    for key in new_contours.keys():
        
        x, y, w, h = new_contours[key]
        
        # Extract the letter from the original image with a 2-pixel margin around the edge
        #letter_image = pad[y - 2:y + h + 2, x - 2:x + w + 2]
        letter_image = pad[y:y + h, x:x + w]
        
        #resize
        l_I2828 = image_resize(letter_image, 28, 28)
        #l_I2828 = imutils.resize(letter_image, width=28, height=28)
        #l_I2828 = cv2.resize(letter_image, (28, 28), interpolation=cv2.INTER_AREA)
                #interpolation=cv2.INTER_AREA ==> for decimating/ resizing image default INTER_LINEAR 
                # https://www.geeksforgeeks.org/image-resizing-using-opencv-python/
                #https://docs.opencv.org/3.4.0/da/d54/group__imgproc__transform.html#gga5bb5a1fea74ea38e1a5445ca803ff121acf959dca2480cc694ca016b81b442ceb
                
        
        #Normalisation/ reshape
        K = l_I2828/255.0  # Normalisation 
        K = K.reshape(-1,28,28,1)
        
        #prediction
        pred = model.predict(K)
        
        #append list 
        letter = onehot.inverse_transform(pred)[0]
        
        #correct format
        if letter < 10:
            captcha += "0" + str( letter )
        else:
            captcha += str( letter ) 
    
    print(captcha)
    final_pred.append(int(captcha))
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