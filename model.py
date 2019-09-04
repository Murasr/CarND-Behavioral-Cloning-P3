import os
import csv
import cv2
import math
import numpy as np
import sklearn
from scipy import ndimage
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Cropping2D, Conv2D, Flatten, Dense, Lambda, Dropout, Activation, MaxPooling2D

#columns : ['center', 'left','right','steering','throttle','brake','speed']
def getLinesFromCSVFile(csvFilePath):
    lines = []
    with open(csvFilePath) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines

def generator(imgDirec, samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # Centre image
                name = imgDirec + batch_sample[0].split('\\')[-1]
                center_image = ndimage.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                # Centre image flipped
                center_image_flipped = np.fliplr(center_image)
                center_angle_flipped = center_angle * -1.0
                images.append(center_image_flipped)
                angles.append(center_angle_flipped)
                
                #consider left and right images only for the images which are at the center (ie., abs(steering angle) < 0.5)
                if (np.abs(center_angle) <= 0.4):
                    #25 degree rotation is mapped to range (0 to 1) of steering angle. 
                    # considering the left and right cameras are mounted at 1.2m from the centre.
                    #vehicle has to reach the centre in 10m, the correction factor is obtained as arctan2(1.2/10) * 180 / 3.14 * 1 / 25
                    correction_factor = 0.272
                    #Left image
                    left_name = imgDirec + batch_sample[1].split('\\')[-1]
                    left_image = ndimage.imread(left_name)
                    left_angle = center_angle + correction_factor
                    images.append(left_image)
                    angles.append(left_angle)
                    # left image flipped
                    left_image_flipped = np.fliplr(left_image)
                    left_angle_flipped = left_angle * -1.0
                    images.append(left_image_flipped)
                    angles.append(left_angle_flipped)

                    #Right image
                    right_name = imgDirec + batch_sample[2].split('\\')[-1]
                    right_image = ndimage.imread(right_name)
                    right_angle = center_angle - correction_factor
                    images.append(right_image)
                    angles.append(right_angle)
                    # right image flipped
                    right_image_flipped = np.fliplr(right_image)
                    right_angle_flipped = right_angle * -1.0
                    images.append(right_image_flipped)
                    angles.append(right_angle_flipped)
                

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            
def trainAndSaveModel(train_generator, validation_generator, batch_size):
    row, col, ch = 160, 320, 3
    top_cropping, bottom_cropping = 50, 0

    model = Sequential()
    model.add(Cropping2D(cropping=((top_cropping,bottom_cropping), (0,0)), input_shape=(row,col,ch)))
    row = row - top_cropping - bottom_cropping

    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(Lambda(lambda x: (x/255.0) - 0.5,
            input_shape=(row, col, ch),
            output_shape=(row, col, ch)))
    
    model.add(Conv2D(24, (5,5), strides=(2,2), padding='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(36, (5,5), strides=(2,2), padding='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(48, (5,5), strides=(2,2), padding='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3,3), strides=(1,1), padding='valid'))
    #model.add(MaxPooling2D((2, 2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    
    model.add(Dense(50))
    model.add(Activation('relu'))
    
    model.add(Dense(10))
    model.add(Activation('relu'))
    
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    #TODO: change steps as per data augmentation
    model.fit_generator(train_generator, \
                steps_per_epoch=math.ceil(len(train_samples)/batch_size), \
                validation_data=validation_generator, \
                validation_steps=math.ceil(len(validation_samples)/batch_size), \
                epochs=5, verbose=1)

    model.save('model.h5')
            

        
#Actual Work is done here
line_samples = getLinesFromCSVFile('./data_generated_1/data_generated/driving_log.csv')

print('Total number of original samples: ', len(line_samples))

train_samples, validation_samples = train_test_split(line_samples, test_size=0.2)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator('./data_generated_1/data_generated/IMG/', train_samples, batch_size=batch_size)
validation_generator = generator('./data_generated_1/data_generated/IMG/', validation_samples, batch_size=batch_size)

trainAndSaveModel(train_generator, validation_generator, batch_size)

