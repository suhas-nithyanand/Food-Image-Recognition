# Food-Image-Recognition

Food is my first love and I love spicy food.
So for my first machine learning project, I decided to work on Food image recognition.

Dataset selection:
As with any machine learning problem, a challenging huge dataset is crucial for the experiment.
I have selected Food-101 datset for my project.
You can download the project from their site at: https://www.vision.ee.ethz.ch/datasets_extra/food-101/

Step1:

Installing Caffe: http://caffe.berkeleyvision.org/installation.html 
Follow the instructions in the above link to setup caffe environment.
After finishing all the make commands for caffe, also install requirements for python and perform make pycaffe.

Using the trained deep learning model will speed up our training process.
Download the googlenet trained model from this link: https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet
Place the in models/googlenet folder. 

Step2:

I used rename.py to rename all the images with foodname + Image number counter for visual aid.

Use this command to get all image names ( address on your machine) in a single text file.
find `pwd`/{images diectory} -type f -exec echo {} \; > images.txt.
This file will be used later an input to caffe_feature_extractor.py

Step3:

Now extract high level image features from googlenet model using caffe_feature_extractor.py. 
This python file takes input.txt (image addresses obtained from previous step) and 
output.txt (empty text file where extracted image vectors will be stored) as arguments. 
Change all the paths relative to your caffe installations.

Step4:

Now install Scikit learn python library in your system.
Now we have separate the training and test dataset. 
An important step here is to randomize the data, so that we have almost equal distributions of all classes in the test and training dataset.
For my experiment, I have split my training and test data in the ratio 90:10. But you could experiment with other ratios.

Step5:

Although, I didn't spend a lot of time tuning the SVM constants, It's an important step to get good accuracy.
Finally we can evaluate the prediction by comparing it with the output.
Accuracy and F-Score can be evaluated using Scikit library functions. 
I obtained an accuracy of ~0.55 with the SVM constants given in code. 
As a logical next step, I plan to train the dataset using a Convolutional Neural Network and obtain classifcation results.
