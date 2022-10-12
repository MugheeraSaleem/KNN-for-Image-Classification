# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 13:04:01 2022

@author: MugheeraSaleem
"""
import numpy as np
import cv2
import os.path
from KNearestNeighbor_class import KNearestNeighbor


Dataset_path = 'E:/Masters/Semester 3/DEEP LEARNING/KNN For Image Classification/Horses_Duck_Dataset'
Prediction_files_path = 'E:/Masters/Semester 3/DEEP LEARNING/KNN For Image Classification/Horses_Duck_Dataset/Prediction Images'
labels = ["Horse","Duck" ]


def pre_processing(image_path):
    '''
    Images are first converted from RGB to grey and then the images are reduced in size. Then these reducded images
    are normalized by dividing them with 255. 
    
    Input:
    - image_path: This is the path of the image that is to be pre_processed.
    
    Returns:
    - img_pred: The processed image, it is first converted into grey and then resized to 100x100 and then normalized.
    
    '''
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_pred = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
    img_pred = np.asarray(img_pred)
    img_pred = img_pred / 255
    return img_pred

def load_image_files(container_path):
    '''
    All the image files that are inside the given directory are first pre_processed using the pre_processing function.
    Then the processed images are appended in an array named train_img.
    
    Input:
    - Container_path: Path of the directory which conatains all the image files. 
    Returns:
    - X: This array contains all the images appended in it in the form of a array.
    '''
    train_img = []
    for file in os.listdir(container_path):
        if (os.path.isfile(container_path + "/" + file)):
            img_pred = pre_processing(container_path + "/" + file)
            train_img.append(img_pred)
    X = np.array(train_img)
    return X


###############################################################################
###################### Creating Features from Images ##########################
###############################################################################


X = []
X = load_image_files(Dataset_path)

y0 = np.zeros(10)
#10 is the number of Horses in X
y1 = np.ones(10)
#10 is the number of Ducks in X
y = []
y = np.concatenate((y0,y1), axis=0)


X_train = X
y_train = y

print("X_train: "+str(X_train.shape))
print("y_train: "+str(y_train.shape))


# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))

print("X_train: "+str(X_train.shape))
print("y_train: "+str(y_train.shape))


###############################################################################
########################### Prediction Function ###############################
###############################################################################


def prediction_function(image_name):
    '''
    Predict a class against an input image. All the preprocessing is done in this function. 
    
    Inputs:
    - Image_name: THis is the name of the file on which the class has to be preicted. 
    '''
    img_pred = pre_processing(Prediction_files_path +'/'+ image_name)
    img_pred = np.reshape(img_pred, (1, img_pred.shape[0]*img_pred.shape[1]))
    classifier = KNearestNeighbor()
    classifier.train(X_train, y_train)
    dists_L2 = classifier.compute_distances(img_pred)
    y_test_pred = classifier.predict_labels(dists_L2, k=3)
    print('Predicted '+image_name + ' as a ' + labels[int(y_test_pred)])


print("Predicting custom images")
for names in os.listdir(Prediction_files_path):    
    prediction_function(names)
