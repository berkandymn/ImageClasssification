#%%
#kütüphaneler
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import pathlib

import sklearn.metrics as mt
import tensorflow as tf

from PIL import Image


dataSet='dataSet/'

allImages = pathlib.Path(dataSet)
imageDs = len(list(allImages.glob('*/*.jpg'))) 

image=list(allImages.glob('*/*.jpg'))
if image:
    frame=image[0]
    with Image.open(frame) as img:
        width, height = img.size
        print(f"Resim: {frame}, Boyut: {width}x{height}")
        frame=image[299]
    with Image.open(frame) as img:
        width, height = img.size
        print(f"Resim: {frame}, Boyut: {width}x{height}")
        frame=image[1190]
    with Image.open(frame) as img:
        width, height = img.size
        print(f"Resim: {frame}, Boyut: {width}x{height}")

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
    for i in range(30): 
        plt.rcParams.update({'text.color': "red",
                     'axes.labelcolor': "green"})
        ax = plt.subplot(6, 5, i + 1) 
        plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
        plt.imshow(images[i].numpy().astype("uint8")) 
        plt.title(classes[labels[i]]) 
        
        plt.axis("off") 
        
classesNum = len(classes) 

"""  #kendi oluşturduğumz model
MLmodel = tf.keras.Sequential([ 
    tf.keras.layers.Rescaling(1./255, input_shape=(180,180, 3)), 
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'), 
    tf.keras.layers.MaxPooling2D(), 
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'), 
    tf.keras.layers.MaxPooling2D(), 
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'), 
    tf.keras.layers.MaxPooling2D(), 
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(128, activation='relu'), 
    tf.keras.layers.Dense(classesNum) 
]) """

MLmodel=tf.keras.applications.MobileNetV3Large(
    input_shape=None,
    alpha=1.0,
    minimalistic=False,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    classes=1000,
    pooling=None,
    dropout_rate=0.2,
    classifier_activation="softmax",
    include_preprocessing=True,
)

MLmodel.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy( 
                  from_logits=True), 
              metrics=['accuracy']) 
MLmodel.summary() 

epochs=10
history = MLmodel.fit( 
  trainDs, 
  validation_data=valDs, 
  epochs=epochs 
) 

#Accuracy 
acc = history.history['accuracy'] 
val_acc = history.history['val_accuracy'] 
  
#loss 
loss = history.history['loss'] 
val_loss = history.history['val_loss'] 
  
#epochs  
epochs_range = range(epochs) 
  
# eğitim sonucu grafikleri
plt.figure(figsize=(8, 8)) 
plt.subplot(1, 2, 1) 
plt.plot(epochs_range, acc, label='Training Accuracy') 
plt.plot(epochs_range, val_acc, label='Validation Accuracy') 
plt.legend(loc='lower right') 
plt.title('Training and Validation Accuracy') 
  

plt.subplot(1, 2, 2) 
plt.plot(epochs_range, loss, label='Training Loss') 
plt.plot(epochs_range, val_loss, label='Validation Loss') 
plt.legend(loc='upper right') 
plt.title('Training and Validation Loss') 
plt.show() 

# Tahminler
y_pred_probs = MLmodel.predict(valDs)
y_pred = np.argmax(y_pred_probs, axis=1)

# Gerçek etiketler
y_true = np.concatenate([y for x, y in valDs], axis=0)

# Karışıklık matrisini hesaplama
conf_matrix = mt.confusion_matrix(y_true, y_pred)

# F1 skoru ve geri çağırma değerlerini hesapla
f1 = mt.f1_score(y_true, y_pred, average='weighted')
recall = mt.recall_score(y_true, y_pred, average='weighted')

# Sonuçları yazdırma
print(f"F1 Score: {f1}")
print(f"Recall: {recall}")
print("Confusion Matrix:")
print(conf_matrix)

# Karışıklık matrisini görselleştirme
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()