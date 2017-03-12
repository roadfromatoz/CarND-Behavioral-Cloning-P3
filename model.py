import csv
import cv2
import os
import numpy as np

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

import pickle
import tensorflow as tf
import cv2
import numpy as np
import sklearn

csv_dir = './data_capture'
lines = []
using_generator = True

# save the csv lines
with open(os.path.join(csv_dir, 'driving_log.csv')) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# option 1: do not use generator (excessive memory use, slow as data size grow)
if not using_generator:
    if not os.path.isfile('data.pickle'):
        
        images = []
        steers = []
        
        for line in lines:
            try:
                test = float(line[3])
            except:
                # skip first header
                continue
            center_img_path = line[0]
            filename = line[0].split('/')[-1]
            center_img_path = os.path.join(csv_dir, 'IMG', filename)
            image = cv2.imread(center_img_path)
            assert not image is None
            images.append(image)
            steers.append(float(line[3]))
            images.append(cv2.flip(image, 1))
            steers.append(-float(line[3]))
        
        X_train = np.array(images)
        y_train = np.array(steers)

        pickle.dump({'X_train': X_train, 'y_train': y_train}, open('data.pickle', 'wb'))
    else:
        data = pickle.load(open('data.pickle', 'rb'))
        X_train = data['X_train']
        y_train = data['y_train']

# option 2: use generator
else:
    from sklearn.model_selection import train_test_split
    train_samples, validation_samples = train_test_split(lines, test_size=0.2)
    train_samples_new = []
    for line in train_samples:
        train_samples_new.append(line + ['normal_center',])
    # flip image to prevent weighing more on left turns
    for line in train_samples:
        train_samples_new.append(line + ['flipped_center',])
    # use side images to generalize better
    for line in train_samples:
        train_samples_new.append(line + ['normal_left',])
    for line in train_samples:
        train_samples_new.append(line + ['flipped_left',])
    for line in train_samples:
        train_samples_new.append(line + ['normal_right',])
    for line in train_samples:
        train_samples_new.append(line + ['flipped_right',])
    # apply additional offset because side cameras are biased more to side
    side_camera_offset = 0.10
        
    def generator(samples_in, batch_size=64, post_process=False):
        num_samples = len(samples_in)
        while 1: # Loop forever so the generator never terminates
            samples = sklearn.utils.shuffle(samples_in)
            for offset in range(0, num_samples, int(batch_size)):
                batch_samples = samples[offset:offset+int(batch_size)]
                images = []
                angles = []
                for batch_sample in batch_samples:
                    if post_process:
                        image_type, camera  = batch_sample[-1].split('_')
                    else:
                        image_type, camera = ('normal', 'center')
                    if camera == 'center':
                        filename = batch_sample[0].split('/')[-1]
                        angle = float(batch_sample[3])
                    elif camera == 'left':
                        filename = batch_sample[1].split('/')[-1]
                        angle = float(batch_sample[3]) + side_camera_offset
                    elif camera == 'right':
                        filename = batch_sample[2].split('/')[-1]
                        angle = float(batch_sample[3]) - side_camera_offset
                    img_path = os.path.join(csv_dir, 'IMG', filename)
                    image = cv2.imread(img_path)
                    assert not image is None
                    if image_type == 'normal':
                        images.append(image)
                        angles.append(angle)
                    else:
                        assert image_type == 'flipped'
                        images.append(cv2.flip(image, 1))
                        angles.append(-angle)
    
                X_train = np.array(images)
                y_train = np.array(angles)
                yield sklearn.utils.shuffle(X_train, y_train)

    # compile and train the model using the generator function
    train_generator = generator(train_samples_new, batch_size=64, post_process=True)
    validation_generator = generator(validation_samples, batch_size=64)

    
with tf.device('/cpu:0'):
    ## model I: 1 layer regression
    # model = Sequential()
    # model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
    # model.add(Flatten(input_shape=(160, 320, 3)))
    # model.add(Dense(1))

    ## model II: LeNet
    # model = Sequential()
    # model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
    # model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    # model.add(Convolution2D(6, 5, 5, activation='relu', input_shape=(160, 320, 3)))    
    # model.add(MaxPooling2D())
    # model.add(Convolution2D(16, 5, 5, activation='relu'))
    # model.add(MaxPooling2D())
    # model.add(Flatten())
    # model.add(Dense(120))
    # model.add(Dense(84))
    # model.add(Dense(1))

    ## model III: NVidia
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mse', optimizer=adam)
    if not using_generator:
        model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch = 3)
    else:
        filepath="models-nvidia-0.10/weights-improvement-{epoch:02d}.hdf5"
        # save models per epoch
        checklist = ModelCheckpoint(filepath, verbose=1, period=1)
        model.fit_generator(train_generator,
                            samples_per_epoch= len(train_samples_new),
                            validation_data=validation_generator,
                            nb_val_samples=len(validation_samples),
                            nb_epoch=15,
                            callbacks=[checklist])

model.save('models-nvidia-0.10.h5')
