#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/RandomSamples.png "Random Samples"
[image2]: ./examples/LabelHistograms.png "Label Histograms"
[image3]: ./examples/TrafficSignsExamples.png "Traffic Sign Examples"
[image4]: ./examples/NormalizedImages.png "Normalized Image Examples"
[image5]: ./examples/NewImages.png "New Images"
[image6]: ./examples/NewImagesClassification.png "New Images Classification"
[image7]: ./examples/FeatureMaps.png "Feature Maps"

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

This is a writeup summary of the implementation of the traffic sign classifier project and here is a link to my [project code](https://github.com/lluiscanet/CarND-Traffic-Sign-Classifier-Project)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of testing examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32)
* Image channels = 3
* Number of classes = 43

####2. Include an exploratory visualization of the dataset.

First of all, we visualize a random image from each of the sets (training, validation and testing). 
![alt text][image1]

Then, we want to understand the different classes available in the dataset so we are getting a random samples of each of the image classes available:
![alt text][image3]


We also visualized the several bar charts for the training, validation and testing data sets for each of the available classes.
![alt text][image2]


###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

We were able to achieve the desired results by doing some simple pre-processing on the images. Before feeding the images into the CNN network we are normalizing the values for each of the color channels. Since the max value for each of the pixel is 255, we are multiplying the pixel value by 2 and then dividing by 255 and substracting 1:
`norm_imgs = images/255*2-1`

This normalizes all values on the range between -1 and 1. We could have added more training data by rotating some of the images or adding noise to them. We could have also filetered out irrelevant color values, such as green pixels. Another improvement could be to add another layer to the input images that includes the grayscale values. However, just by adding the previously described normalization we were able to achieve the desired results.

Here are how the normalized images look like

![alt text][image4]

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.


The final architecture is based on the LeNet architecture with the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	  		| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	  		| 2x2 stride,  outputs 5x5x16 					|
| Fully connected		| Input 400 elements, output elements 120		|
| RELU					|												|
| Fully connected		| Input 120 elements, output elements 84		|
| RELU					|												|
| Dropout				| Keep Probability 0.7							|
| Fully connected		| Input 84 elements, output elements 43			|
| RELU					|												|
| Softmax				| Only used to obtain output probabilities   	|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model I used a batch size of 128 and 10 epochs. The mean and standard deviation for the initialization of the Weight and Biases is 0 and 0.1 respectively. The learning rate chosen is 0.001 and the keep probability for the dropout layer is 0.7

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The final results for each of the iterations are the following:
```

EPOCH 1 ...
Training Accuracy = 0.487
Validation Accuracy = 0.781

EPOCH 2 ...
Training Accuracy = 0.835
Validation Accuracy = 0.875

EPOCH 3 ...
Training Accuracy = 0.909
Validation Accuracy = 0.906

EPOCH 4 ...
Training Accuracy = 0.940
Validation Accuracy = 0.901

EPOCH 5 ...
Training Accuracy = 0.956
Validation Accuracy = 0.920

EPOCH 6 ...
Training Accuracy = 0.963
Validation Accuracy = 0.927

EPOCH 7 ...
Training Accuracy = 0.969
Validation Accuracy = 0.924

EPOCH 8 ...
Training Accuracy = 0.975
Validation Accuracy = 0.923

EPOCH 9 ...
Training Accuracy = 0.979
Validation Accuracy = 0.932

EPOCH 10 ...
Training Accuracy = 0.981
Validation Accuracy = 0.937

```

Therefore, the final training, validation and testing accuracies are:
```
Training Accuracy = 0.981
Validation Accuracy = 0.937
Test Accuracy = 0.927
```
Originally, we tested with the same network architecture and the same parameters but we did not have a dropout layer. We observed that there was a signficant discrepancy between the training accuracy and the validation accuracy, this later beeing less than 93% which is our goal. This was telling us that the model was overfitting so adding a dropout layer could resolve the problem. We added the dropout layer with a keep probability of 0.5 but affected the training accuracy significantly and we were not able to reach the validation accuracy desired. By increasing the keep probability to 0.7 we were able to achieve the results posted above.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

We were able to find the following images on the web:

![alt text][image5]

We had to crop the images and reduce their size to 32x32. Most of them are clear images, the only two that can cause some issues are the speed limit one and the no-entry one due to slight rotation and perspective.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:
![alt text][image6]


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. Although the test accuracy is much higher, we do not have enough samples to get a reliable estimate of the real accuracy with 5 images. However, it is reasonalbe that we are getting 1 image incorrectly classified out of 5.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)
Here are the top results for each of the images based on the softmax score:
```
Top softmax for image1.jpg:
	Label: 14-Stop --> Softmax: 1.000
	Label: 1-Speed limit (30km/h) --> Softmax: 0.000
	Label: 25-Road work --> Softmax: 0.000
	Label: 3-Speed limit (60km/h) --> Softmax: 0.000
	Label: 2-Speed limit (50km/h) --> Softmax: 0.000
Top softmax for image2.jpg:
	Label: 33-Turn right ahead --> Softmax: 1.000
	Label: 39-Keep left --> Softmax: 0.000
	Label: 40-Roundabout mandatory --> Softmax: 0.000
	Label: 35-Ahead only --> Softmax: 0.000
	Label: 37-Go straight or left --> Softmax: 0.000
Top softmax for image3.jpg:
	Label: 13-Yield --> Softmax: 1.000
	Label: 35-Ahead only --> Softmax: 0.000
	Label: 9-No passing --> Softmax: 0.000
	Label: 3-Speed limit (60km/h) --> Softmax: 0.000
	Label: 15-No vehicles --> Softmax: 0.000
Top softmax for image4.jpg:
	Label: 14-Stop --> Softmax: 0.580
	Label: 25-Road work --> Softmax: 0.419
	Label: 9-No passing --> Softmax: 0.000
	Label: 17-No entry --> Softmax: 0.000
	Label: 29-Bicycles crossing --> Softmax: 0.000
Top softmax for image5.jpg:
	Label: 7-Speed limit (100km/h) --> Softmax: 1.000
	Label: 5-Speed limit (80km/h) --> Softmax: 0.000
	Label: 10-No passing for vehicles over 3.5 metric tons --> Softmax: 0.000
	Label: 2-Speed limit (50km/h) --> Softmax: 0.000
	Label: 16-Vehicles over 3.5 metric tons prohibited --> Softmax: 0.000
```

As we can observe, all the correctly classified images have a very confident score. The only incorrectly classified one is the one with confidence score lower than 1.0. The correct response is the third option.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
This is the visualization of the weights on the convolutional layer 2. They are not very interpretable since they are a higher level abstraction.
![alt text][image7]

