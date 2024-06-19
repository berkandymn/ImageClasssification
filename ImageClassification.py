#%%
#kütüphaneler
import pandas as pd
import numpy as np
import PIL 
import seaborn as sns 
import matplotlib.pyplot as plt
import pprint
import pathlib


import sklearn.metrics as mt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_val_predict

from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer, make_column_selector

#modeller
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

import tensorflow as tf



dataSet='dataSet/'

allImages = pathlib.Path(dataSet)
imageDs = len(list(allImages.glob('*/*.jpg'))) 

trainDs = tf.keras.utils.image_dataset_from_directory( 
    allImages, 
    validation_split=0.2, 
    subset="training", 
    seed=123, 
    image_size=(180, 180), 
    batch_size=32) 

valDs = tf.keras.utils.image_dataset_from_directory( 
    allImages, 
    validation_split=0.2, 
    subset="validation", 
    seed=123, 
    image_size=(180,180), 
    batch_size=32) 

classes = trainDs.class_names 
print(classes)

plt.figure(figsize=(10,10))

for images, labels in trainDs.take(1): 
    for i in range(25): 
        ax = plt.subplot(5, 5, i + 1) 
        plt.imshow(images[i].numpy().astype("uint8")) 
        plt.title(classes[labels[i]]) 
        plt.axis("off") 