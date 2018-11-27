# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/training_example.jpg "Training Example"
[image2]: ./examples/cropped_yuv.jpg "Cropped YUV Image"
[image3]: ./examples/before_after_image_flip.jpg "Image Flip"
[image4]: ./examples/before_after_image_shifting.jpg "Image Shift"
[image5]: ./examples/before_after_image_shadow.jpg "Image Shadow"
[image6]: ./examples/before_after_image_brightness.jpg "Image Brightness"
[image7]: ./examples/before_after_augmentation.jpg "Image Augmentation"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

In this project, I used a deep neural network (built with Keras) to clone car driving behavior.

The dataset used to train the network is generated from Udacity's Self-Driving Car Simulator, and it consists of images taken from three different camera angles (Center - Left - Right), in addition to the steering angle, throttle, brake, and speed during each frame.


![alt text][image1]

## **Pipeline architecture:** ##

- Data Loading.
- Data Augmentation.
- Data Preprocessing.
- Model Architecture.
- Model Training and Evaluation.
- Model Testing on the simulator.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

In the model architecture, I will design and implement a deep learning model that can clone the vehicle's behavior. I'll use a convolutional neural network (CNN) to map raw pixels from a single front-facing camera directly to steering commands.

I'll use the ConvNet from NVIDIA's paper End to End Learning for Self-Driving Cars, which has been proven to work in this problem domain.


- I've splitted the data into 80% training set and 20% validation set to measure the performance after each epoch.
- I used Mean Squared Error (MSE) as a loss function to measure how close the model predicts to the given steering angle for each input frame.
- I used the Adaptive Moment Estimation (Adam) Algorithm minimize to the loss function. Adam is an optimization algorithm introduced by D. Kingma and J. Lei Ba in a 2015 paper named Adam: A Method for Stochastic Optimization. Adam algorithm computes adaptive learning rates for each parameter. In addition to storing an exponentially decaying average of past squared gradients like Adadelta and RMSprop algorithms, Adam also keeps an exponentially decaying average of past gradients mtmt, similar to momentum algorithm, which in turn produce better results.
- I used ModelCheckpoint from Keras to check the validation loss after each epoch and save the model only if the validation loss reduced.


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py line 350). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 417).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. The data provided by Udacity was used. The simulator provides three different images: center, left and right cameras. Each image was used to train the model.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to try the LeNet](http://yann.lecun.com/exdb/lenet/) model with three epochs and the training data provided by Udacity. On the first track, the car went straight to the lake. I needed to do some pre-processing. A (model.py line 344) `Lambda` layer was introduced to normalize the input images to zero means.

The second step was to use a more powerful model: [nVidia Autonomous Car Group](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) The only modification was to add a new layer at the end to have a single output as it was required. I also added the dropout layer to avoid overfitting and increased the epochs to 10. This time the car did its first complete track, but there was a place in the track where it passes over the "dashed" line. More data was needed. Augmented the data by adding the same image flipped with a negative angle, added shadow and brightness to the images as shown in the preprocessing steps below. After this process, the car continues to be on track.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 342-357) consisted of a convolution neural network with the following layers and layer sizes ...

A model summary is as follows:

```
Layer (type)                     Output Shape          Param #                  
================================================================

lambda_1 (Lambda)                (None, 160, 320, 3)   0          
________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 90, 320, 3)    0                
_________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 43, 158, 24)   1824      
_________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 20, 77, 36)    21636     
_________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 8, 37, 48)     43248     
_________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 6, 35, 64)     27712     
_________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 4, 33, 64)     36928
_________________________________________________________________     
dropout_1 (Dropout)          	(None, 1, 18, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)              (None, 8448)          	  0         
_________________________________________________________________
dense_1 (Dense)                  (None, 100)           844900    
_________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050      
_________________________________________________________________
dense_3 (Dense)                  (None, 10)            510                
_________________________________________________________________
dense_4 (Dense)                  (None, 1)             11      
_________________________________________________________________  
=================================================================

	Total params: 981,819
	Trainable params: 981,819
	Non-trainable params: 0
```

**Model Evaluation:**
	
	Epoch 	Loss 	Validation Loss
	1/10 	0.0266 	0.0112
	2/10 	0.0212 	0.0106
	3/10 	0.0183 	0.0102
	4/10 	0.0170 	0.0085
	5/10 	0.0165 	0.0085
	6/10 	0.0159 	0.0120
	7/10 	0.0155 	0.0093
	8/10 	0.0156 	0.0102
	9/10 	0.0150 	0.0090
	10/10 	0.0145 	0.0093

#### 3. Creation of the Training Set & Training Process

**Step 1: Data Loading**

The dataset used is provided by Udacity. This dataset contains more than 8,000 frame images taken from the 3 cameras (3 images for each frame), in addition to a csv file with the steering angle, throttle, brake, and speed during each frame.

**Step 2: Data Preprocessing**

- Cropping the image to cut off the sky scene and the car front.
- Resizing the image to (160 * 320), the image size that the model expects.
- Converting the image to the YUV color space.
- Normalizing the images (by dividing image data by 127.5 and subtracting 1.0). As stated in the Model Architecture section, this is to avoid saturation and make gradients work better).

**Step 3: Data Augmentation**

- Adjusting the steering angle of random images.
- Flipping random images horizontally, with steering angle adjustment.
- Shifting (Translating) random images, with steering angle adjustment.
- Adding shadows to random images.
- Altering the brightness of random images.

Image cropping:

![alt text][image2]

Image flipping:

![alt text][image3]

Image translation:

![alt text][image4]

Image shadow:

![alt text][image5]

Image brightness:

![alt text][image6]

Image augmentation:

![alt text][image7] 

The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10. I used an adam optimizer so that manually training the learning rate wasn't necessary.