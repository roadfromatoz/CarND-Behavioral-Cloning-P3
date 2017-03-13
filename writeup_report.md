#**Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./three_camera.png "View from three cameras"
[image2]: ./steering_angle.png "Histogram of steering angles"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I used the Nvidia end-to-end model. Input images are first normalized and cropped to road regions. Then followed by 3 5x5 and 2 3x3 convolution layers, they go through 4 fully connected layers to produce a 1d output which is stearing angle.
(model.py lines 146-158)

####2. Attempts to reduce overfitting in the model

I captured 5-6 laps of data on track1 and 1 lap on track2 under different scenarios: clock-wise/anti-clockwise, center of lane/along side of lane, steering back from road. This helps generalize the training process and avoid overfitting.

On top of that, I flipped images (to avoid biasing on one direction) and used side camera images (more data and generalization). (model.py line 91-114)

####3. Model parameter tuning

The model used a default adam optimizer, so the learning rate was not tuned manually (model.py line 159).

####4. Appropriate training data

This is covered in question 2.

The data consists of cameras from left, center and right, as shown below. You may notice the car is slightly towards the left, therefore yielding a positive steering angle to the right.
![alt text][image1]

The following plot shows a histogram of steering angles. 0 seems to be dominant while a little more left turns (given more laps were collected in anti-clockwise direction). Flipping the image helped in balancing the output and not giving unfair preference to left turns.
![alt text][image2]

###Model Architecture and Training Strategy

My first step was to use a convolution neural network model similar to the Nvidia end-to-end network. It turns out to work well from the very beginning. After 1 epoch, I saw a training error of 0.05 and validation error of 0.03. The final model after 7 epochs was able to steer the car most of the time, except when car is near boundary of lane, the model seems to be less robust.

I went on to collect more data to generalize the training more laps, including more driving close to both lanes, recovering from failures. These all helps in making the model more robust.

Another thing I tried is to use side cameras which brings more data for free. However this does not work well in the beginning because I was using a too-large steering offset to compensate the fact that side cameras are off-center. I tuned that angle based on validation set and found 0.1 to be a good choice.

The final video is attached as `video.mp4`

