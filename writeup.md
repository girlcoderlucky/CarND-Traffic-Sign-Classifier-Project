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

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/grayscale_laplacian_sobel.png "Grayscaling, Laplacian, Sobel"
[image3]: ./examples/canny_edge_dectection.png "Canny Edge Detection"
[image4]: ./examples/Right_turn.png "Traffic Sign 1"
[image5]: ./examples/Yield.png 	"Traffic Sign 2"
[image6]: ./examples/Go_straight_or_right.png "Traffic Sign 3"
[image7]: ./examples/No_entry.png "Traffic Sign 4"
[image8]: ./examples/Speed70.png "Traffic Sign 5"
[image9]: ./examples/Stop.png "Traffic Sign 6"
[image10]: ./examples/normalized.png "Normalization"


---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my project code(https://github.com/girlcoderlucky/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

Here is the summary statistics of the traffic signs data set:

	Number of training examples = 34799
	Number of validation examples = 4410
	Number of testing examples = 12630
	Image data shape = (32, 32, 3)
	Number of classes = 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third and fourth code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. Read signnames.csv and ploted. Here's a histogram showing the distribution inside both the training and the testing set

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale to apply fliters. Applied Laplacian, Sobel filters and Canny edge detection but dataset images are very noisy, none of these filters worked to get good accuracy. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]
![alt text][image3]

As a last, I stuck to color dataset since color channels provide additional information for training and gave much better accuracy than grayscale dataset.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The "traffic-signs-data" dataset include train, test and valid sets. Normalization may help in case of High contrast variation among the images so normailized the training set 

![alt text][image10]

Also rotated the images randomly between (0 to 20 degrees) for 10% of the training set and used "shuffle" to shuffle training set.
	X_train, y_train = shuffle(X_train, y_train)
	
I generated additional dataset by spliting the data into training and validation sets by using train_test_split
	X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
	
To cross validate my model, I did this by evaluating with "test" and additional "valid" dataset.

	validation_accuracy = evaluate(X_test, y_test)
	validation_accuracy = evaluate(X_valid, y_valid)

My final training set had 34799 number of images. My validation set and test set had 4410 and 12630 number of images.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final LeNet model is located in the 7th and 8th cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| Input = 28x28x6. Output = 14x14x6 			|
| Convolution 3x3	    | Output = 10x10x16     						|
| Max pooling			| Input = 10x10x16. Output = 5x5x16     		|
| Flatten				| Input = 5x5x16. Output = 400					|
| Fully connected		| Input = 400 Output = 120						|
| Fully connected		| Input = 120 Output = 84 						|
| Fully connected		| Input = 84 Output = 43						|


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 9th cell and model evaluation is in the 10th cell of the ipython notebook. 

To train the model, I used below parameters. I was training the model on my laptop CPU so couldn't use EPOCHS larger than 50. Model started overfitting for more than 25 EPOCHS so decided to use 20. 
	EPOCHS = 25
	BATCH_SIZE = 128
	rate = 0.001

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 10th cell of the Ipython notebook.

My final model gave validation set accuracy of ~97% accuracy.

If a well known architecture was chosen:
* What architecture was chosen?
Selected LeNet architecture which I learnt in the classroom and was simple to use 
* Why did you believe it would be relevant to the traffic sign application?
LeNet works decently well for image classification which gave ~97% accuracy with test dataset. I would definetely like to try Neural networks for better accuracy.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 Validated the model with "validaton" dataset by spliting the train data into training and validation sets with accuracy of ~97% 
 Test dataset gave accuracy of 90%
 Tested the model with 6 new signs found on the web.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web and converted their size to 32x32x3. 
New test images are way brighter and clearer than the training images and also not sure if all the signs are "German" traffic signs so chances of misprediction is higher.

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook. 
The model was able to correctly guess 4 of the 6 traffic signs, which gives an accuracy of 66.66%. Compared to "test" set accuracy which 87% new images gave ~20% lower accuracy. 
I think new test images are way brighter than the training images and also not sure if I download "German" traffic signs only which may have caused misprediction.

Here are the results of the prediction:

	Actual:Turn right ahead
	Predicted:Turn right ahead

	Actual:Yield
	Predicted:Yield

	Actual:Go straight or right
	Predicted:Go straight or right

	Actual:No entry
	Predicted:No entry

	Actual:Speed limit (70km/h)
	Predicted:Traffic signals

	Actual:Stop
	Predicted:Bicycles crossing


####3. Describe how certain the model is when predicting on each of the six new images by looking at the softmax probablities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 
The code for making predictions on my final model is located in the 15th cell of the Ipython notebook.
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Turn right ahead								| 
| .99     				| Yield 										|
| .99					| Go straight or right							|
| 1.00	      			| No entry						 				|
| 0.00				    | Traffic signals								|
| .69				    | Bicycles crossing								|

Below "Actual" indices 33, 13, 36, 17 matches with softmax probabilities 99+% but mispredicted index 14 is has 30% probablilty, is second in the top five index [29, 14,  5,  3, 13, 22]. May more turning in the training hyperparameters will yeild better prediction.
Actual indices [33, 13, 36, 17, 4, 14]
indices=array([	[33, 26, 25, 23, 11,  7],
				[13, 14, 25, 10, 32,  5],
				[36, 26, 38, 33, 39, 34],
				[17, 14, 30, 38,  0,  1],
				[26, 29,  0,  1,  2,  3],
				[29, 14,  5,  3, 13, 22]]))
