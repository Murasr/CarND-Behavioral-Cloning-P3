

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center_2019_07_30_07_35_12_202.jpg "Center lane driving image"
[image2]: ./examples/center_2019_07_30_07_35_12_202_flipped.jpg "Flipped image"
[image3]: ./examples/recovery_driving_center_2019_07_30_09_50_32_071.jpg "Recovery Driving Image"
[image4]: ./examples/recovery_driving_center_2019_07_30_09_50_32_071_cropped.jpg "Recovery Driving Image Cropped"
[image5]: ./examples/org_data_histogram.png "Original Data Histogram"
[image6]: ./examples/aug_data_histogram.png "Augmented Data Histogram"


### Files Submitted

My project includes the following files:
* model.py - Python script to create and train the model
* model.h5 - Trained CNN model
* drive.py - Pre-included script for driving in autonomous mode
* record2.mp4 - Recorded output of autonomous mode driving
* writeup.md - Report summarizing the results

### Training Data collection
The training data was collected by
1) Two laps of center lane driving
2) Recovery driving from lane boundary to the center of the lane.
	The data is collected only during recovery action and not during driving into lane boundary
3) Additional data for smooth navigation at the sharp turns

### Training Data Augmentation
Based on the histogram of the original training data (located at ./examples/org_data_histogram.png), it is observed that the data set is much biased with steering angle value zero.
The data augmentation is done to increase the dataset samples for steering angle values other than zero.

The training data was augmented by the "Generator" itself. Refer to method "generator" in model.py
1) Flipping (ie., mirror image) the center image and the steering angle is also flipped. 
	The original training track consists mostly left turning curves only. Image flipping is necessary to make the model work for right turning curves also
2) Consider left and right images for certain cases only.
	The left and right images are considered only if the actual steering angle is less than 0.40. 
    This conditional consideration is done to avoid the left and right images when the car is actually steering a curve. Considering these images will make the model to oversteer around the curves.
    The steering angle for left and right images is calculated from center steering angle as below:
    Considering left and right cameras are mounted 1.2m from center camera and the car has to align itself at the center position in next 10ms, the following calculation is done:
    steering_angle_correction = arctan2(1.2/10) * 180 / 3.14
    Mapping the angle range (0 to 25 degree) to factor (0 to 1) => Because an angle of 25 degree rotation is mapped to factor 1 by the simulation.
    ==> steering_angle_correction_factor = arctan2(1.2/10) * 180 / 3.14 * 1/25 = 0.272
    The steering angles for left and right images are calculated as follows:
    ==> left_angle = center_angle + correction_factor
    ==> right_angle = center_angle - correction_factor
3) Do the flipping (ie., mirror image) of the above left and right images and consider them

After the dataset augmentation, the histogram (located at ./examples/aug_data_histogram.png) shows equivalent dataset sizes for zero, +ve, -ve steering angle values. This is good dataset for training.


### Training Data Preprocessing
The training data is preprocessed as follows:
1) Cropping - Crop the top 50 pixels as the information contained does not impact the lane position and the location of car in it
2) Normalization - The image pixel values are normalized to be within the range of -0.5 to 0.5.
The both steps are done in the method "trainAndSaveModel" from lines 80 t0 91
3) Shuffling the data - done by the "Generator" method.

### Training and Validation Data Split
The training and validation data is obtained by splitting the data in the ratio of 80:20.

### Network Model for Training
The model used was derived from "Behaviour cloning model" by Nvidia.
CNN model Layers:
1) Convolution 2D Layer - 5x5 kernel - 24 depth layers - 2x2 stride - valid padding - 110x320x3 Input
2) RELU Activation
3) Convolution 2D Layer - 5x5 kernel - 36 depth layers - 2x2 stride - valid padding
4) RELU Activation
5) Convolution 2D Layer - 5x5 kernel - 48 depth layers - 2x2 strides - valid padding
6) RELU Activation
7) Convolution 2D Layer - 3x3 kernel - 64 depth layers - 1x1 strides - valid padding
8) RELU Activation
9) Convolution 2D Layer - 3x3 kernel - 64 depth layers - 1x1 strides - valid padding
10) RELU Activation
11) Dropout layer of 50% dropping ratio
12) Flatten layer
13) Dropout layer of 50% dropping ratio
14) RELU Activation
15) Dense layer - 100 output
16) Dropout layer of 50% dropping ratio
17) RELU Activation
18) Dense layer - 50 output
19) RELU Activation
20) Dense layer - 10 output
21) RELU Activation
22) Dense layer - 1 output    

Since it is not a classification problem and the goal is to predict the steering wheel, the final output layer provides only one output.
The loss paramter is "MSE" (Mean square error) and optimizer used is "Adam". Adam optimizer has separate learning rate for each weights and adapts them individually as training process proceeds which is advantageous than usual gradient descent optimizer.
The model is run for 5 epochs and a batch size of 32 units

The file model.py makes use of Generator to generate the data on the fly rather than pre-processing complete data and storing them which consumes lot of memory

## Overfitting problem
The model is generalized by following steps:
1) Necessary drop out layers are introduced into the original model (actually 3 drop out layers with 50% drop ratio)
2) Shuffling before training
3) Data augmentation - by creating additional data from left and right images

### Output - Autonomous Mode
The model "model.h5" is able to drive the car in autonomous mode. The result has been recorded into the file "record2.mp4" containing 1 lap.
