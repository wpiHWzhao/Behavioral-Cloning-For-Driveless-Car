import csv
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
import sklearn
from random import shuffle

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D


def readData():
    samples = []

    with open('data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    return samples

def generator(samples, batch_size=32, correction = 0.2):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name_center = 'data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name_center)
                center_angle = float(batch_sample[3])
                assert center_angle is not None, 'Center angle is None!!'
                assert center_image is not None, 'Center image is None!!'
                images.append(center_image)
                angles.append(center_angle)
                images.append(cv2.flip(center_image,1))
                angles.append(center_angle*-1.0)

                name_left = 'data/IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name_left)
                left_angle = float(batch_sample[3])
                assert left_angle is not None, 'Left angle is None'
                assert left_image is not None, 'Left image is None'
                images.append(left_image)
                angles.append(left_angle + correction)
                images.append(cv2.flip(left_image,1))
                angles.append((left_angle+correction)*-1.0)

                name_right = 'data/IMG/' + batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name_right)
                right_angle = float(batch_sample[3])
                assert right_angle is not None, 'Right angle is None'
                assert right_image is not None, 'Right image is None'
                images.append(right_image)
                angles.append(right_angle - correction)
                images.append(cv2.flip(right_image,1))
                angles.append((left_angle-correction)*-1.0)


            X_train = np.array(images)
            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)


def nVaidiaModel():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((50, 20), (0, 0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

samples = readData()

train_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_generator = generator(train_samples, batch_size=32, correction=0.2)
validation_generator = generator(validation_samples,batch_size=32, correction=0.2)

print('Total images: ', len(samples))
print('Train samples', len(train_samples))
print('Validation samples', len(validation_samples))

model = nVaidiaModel()

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, steps_per_epoch= len(train_samples),
                                     validation_data=validation_generator, validation_steps=len(validation_samples),
                                     epochs=3, verbose = 1)



model.save('model.h5')
print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])
