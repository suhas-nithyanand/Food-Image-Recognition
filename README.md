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
This file will be used later an input to caffe_feature_extractor.py
find `pwd`/{images diectory} -type f -exec echo {} \; > images.txt 

Step3:

Now we have to extract high level image features from googlenet model using caffe_feature_extractor.py
Change all the paths relative to your caffe installations.

Step4:
Now install opencv in system

