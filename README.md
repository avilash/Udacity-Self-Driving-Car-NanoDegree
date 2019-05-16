## Udacity Self Driving Car Nanodegree Term 1
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

This repository contains source code for all the projects under the [Udacity Self Driving Car Nanodegree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013) program.

Overview
---
### 1. Finding Lane Lines - Basic
**Summary** - Identify lane lines in a video stream using basic image processing like Edge Detection in specific ROIs.
[CODE](https://github.com/avilash/Udacity-Self-Driving-Car-NanoDegree/tree/master/CarND-LaneLines-P1)

### 2. Finding Lane Lines - Advanced
**Summary** - The project implements finding lanes on a road, measuring its curvature and finding the offset of the vehicle from the lane center. Steps involved
* Distortion correction of images after calibration.
* Using color and gradient thresholds to get a sense of where the lane lines are.
* Perspective transform to get a top view of the lane lines
* A sliding window search for the lane lines
* Finally finding a plot for the lane lines and projecting it back to the original camera perpective

[CODE](https://github.com/avilash/Udacity-Self-Driving-Car-NanoDegree/tree/master/CarND-Advanced-Lane-Lines)

### 3. Traffic Sign Classifier
**Summary** - The project involves building a deep neural net for traffic sign classification. The dataset used if the German Traffic Signs dataset. Involved designing the neural net, preprocessing of images, and using dropout as well as other regularisation techniques to reduce overfitting.
[CODE](https://github.com/avilash/Udacity-Self-Driving-Car-NanoDegree/tree/master/CarND-Traffic-Sign-Classifier-Project)

### 4.  Behavioral Cloning
**Summary** - The project involves trying to clone human driving behaviour. We use camera images and corresponding steering angles collected during human driving in a simulator to train a CNN to predict steering angles from camera images.
[CODE](https://github.com/avilash/Udacity-Self-Driving-Car-NanoDegree/tree/master/CarND-Behavioral-Cloning-P3)

### 5.  Extended Kalman Filter
**Summary** - The project involves the extended Kalman filter in C++. Simulated Lidar and Radar measurements are used to estimate the state of a moving target with error measured as RMSE values against a ground truth.
[CODE](https://github.com/avilash/Udacity-Self-Driving-Car-NanoDegree/tree/master/CarND-Extended-Kalman-Filter-Project)
