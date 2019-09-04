import os
import csv
import math
import numpy as np
import matplotlib.pyplot as plt


#columns : ['center', 'left','right','steering','throttle','brake','speed']
def getLinesFromCSVFile(csvFilePath):
    lines = []
    with open(csvFilePath) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines

### Data exploration visualization code goes here.
plt.switch_backend('Agg')

#Actual Work is done here
line_samples = getLinesFromCSVFile('./data_generated_1/data_generated/driving_log.csv')
org_angles = [] # original dataset angles
aug_angles = [] # augmented dataset angles
for line in line_samples:
    # center image steering wheel angle
    center_angle = float(line[3])
    org_angles.append(center_angle)
    
    aug_angles.append(center_angle)
    #center image flipped case
    aug_angles.append(center_angle * -1.0)
    
    #consider left and right images only for the images which are at the center (ie., abs(steering angle) < 0.5)
    if (np.abs(center_angle) <= 0.4):
        #25 degree rotation is mapped to range (0 to 1) of steering angle. 
        # considering the left and right cameras are mounted at 1.2m from the centre.
        #vehicle has to reach the centre in 10m, the correction factor is obtained as arctan2(1.2/10) * 180 / 3.14 * 1 / 25
        correction_factor = 0.272
        #left image steering angle factor
        left_angle = center_angle + correction_factor
        aug_angles.append(left_angle)
        # left image flipped case
        aug_angles.append(left_angle * -1.0)

        #right image steering angle factor
        right_angle = center_angle - correction_factor
        aug_angles.append(right_angle)
        # right image flipped case
        aug_angles.append(right_angle * -1.0)
                
#Save the histogram of original data
plt.hist(org_angles, bins=20)
plt.xlabel('Original Data Steering angle')
plt.ylabel('count')
plt.savefig('./examples/org_data_histogram.png')

#Save the histogram of augmented data
plt.hist(aug_angles, bins=20)
plt.xlabel('Augmented Data Steering angle')
plt.ylabel('count')
plt.savefig('./examples/aug_data_histogram.png')