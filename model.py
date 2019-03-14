import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sklearn
import random

lines = []
with open('./Data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def generator(samples, batch_size):
    num_samples = len(samples)
    
    while 1: # Loop forever so the generator never terminates
      
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
        
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
               # i = random.choice(range(3))
                for i in range(3):
                    name = './Data/IMG/'+batch_sample[i].split('\\')[-1]
                      
                    img = cv2.imread(name)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)               
                
                    if i == 0:  
                        
                        angle = float(batch_sample[3])
                    elif i == 1:
                    
                        angle = float(batch_sample[3]) + 0.2
                    else:
                    
                        angle = float(batch_sample[3]) -0.2
                
                images.append(img)
                image_flipped = np.fliplr(img)
                images.append(image_flipped)
                
                angles.append(angle)
                measurement_flipped = -angle
                angles.append(measurement_flipped)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
      
            
            yield sklearn.utils.shuffle(X_train, y_train)
            
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=8)
validation_generator = generator(validation_samples, batch_size=8)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5,
        input_shape = (160,320,3)))
model.add(Cropping2D(cropping = ((75,20),(0,0))))
          
model.add(Convolution2D(24,5,5, subsample = (2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample = (2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample = (2,2), activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Dropout(0.2))
model.add(Flatten()) 
model.add(Dense(640))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')

history_object = model.fit_generator(train_generator, samples_per_epoch =
    len(train_samples), validation_data = 
    validation_generator,
    nb_val_samples = len(validation_samples), 
    nb_epoch=3, verbose=1)
    
model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()