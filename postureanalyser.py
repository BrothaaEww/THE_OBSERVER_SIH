import os
import glob
import random
import numpy as np
import pandas as pd

# import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from tqdm import tqdm

from PIL import Image

from keras.utils import to_categorical

import seaborn as sns
import matplotlib.image as img
import matplotlib.pyplot as plt
import pickle 

train_csv = pd.read_csv("F:/Coding/Python_VS_Code/sec_proj/har/Human Action Recognition/Training_set.csv")
test_csv = pd.read_csv("F:/Coding/Python_VS_Code/sec_proj/har/Human Action Recognition/Testing_set.csv")

train_fol = glob.glob("F:/Coding/Python_VS_Code/sec_proj/har/Human Action Recognition/train/*") 
test_fol = glob.glob("F:/Coding/Python_VS_Code/sec_proj/har/Human Action Recognition/test/*")

print(train_csv.label.value_counts())

print(train_csv)

filename = train_csv['filename']
situation = train_csv['label']

imgg = "Image_{}.jpg".format(1)
train = "F:/Coding/Python_VS_Code/sec_proj/har/Human Action Recognition/train/"
testImage = img.imread(train + imgg)
plt.imshow(testImage)
plt.title("{}".format(train_csv.loc[train_csv['filename'] == "{}".format(imgg), 'label'].item()))
plt.show()

#preprocessing

img_data = []
img_label = []
length = len(train_fol)
for i in (range(len(train_fol)-1)):
    t = 'F:/Coding/Python_VS_Code/sec_proj/har/Human Action Recognition/train/' + filename[i]    
    temp_img = Image.open(t)
    img_data.append(np.asarray(temp_img.resize((160,160))))
    img_label.append(situation[i])
    print(f"processed {i}")
    
inp_shape = (160, 160,3)
arr = img_data
arr = np.asarray(arr)
print(type(arr))

y_train = to_categorical(np.asarray(train_csv['label'].factorize()[0]))
print(y_train[0])

vgg_model = Sequential()

pretrained_model= tf.keras.applications.VGG16(include_top=False,
                   input_shape=(160,160,3),
                   pooling='avg',classes=15,
                   weights='imagenet')

for layer in pretrained_model.layers:
        layer.trainable=False

vgg_model.add(pretrained_model)
vgg_model.add(Flatten())
vgg_model.add(Dense(512, activation='relu'))
vgg_model.add(Dense(15, activation='softmax'))

vgg_model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
print(vgg_model.summary())
es = EarlyStopping(monitor='val_accuracy', mode='max', patience=5,  restore_best_weights=True)
history = vgg_model.fit(arr,y_train, epochs=40,callbacks=[es])
with open('vgg_model.pkl', 'wb') as f:
    pickle.dump(vgg_model, f)

vgg_model.save_weights("model.h5")

def test_predict(test_image):
    situation=["sitting","using_laptop","hugging","sleeping","drinking",
           "clapping","dancing","cycling","calling","laughing"
          ,"eating","fighting","listening_to_music","running","texting"]
    image = Image.open(test_image)
    input_img = np.asarray(image.resize((160,160)))
    result = vgg_model.predict(np.asarray([input_img]))

    itemindex = np.where(result==np.max(result))
    prediction = itemindex[1][0]
    print("probability: "+str(np.max(result)*100) + "%\nPredicted class : ", situation[prediction])

    image = img.imread(test_image)
    plt.imshow(image)
    
test_predict('F:/Coding/Python_VS_Code/sec_proj/har/Human Action Recognition/test/Image_50.jpg')