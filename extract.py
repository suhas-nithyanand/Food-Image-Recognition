import math
#from sklearn import svm
import numpy as np
import cv2
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix



food_list = ['apple_pie','chicken_wings','french_onion_soup','fried_rice','greek_salad','hot_dog','omelette','onion_rings','samosa','sushi']

'''10000 image vectors'''

lines = open('output_vectors.txt').read().splitlines()

vectors_list = []
each_feature = []
for l in lines:
	vector = l
	vals= vector.split()
	each_feature = []
	for v in vals:
		each_feature.append(v)
	vectors_list.append(each_feature)
#print len(vectors)

vectors = np.asarray(vectors_list)

'''input labels'''
labels = []
output_label_list = []

labels = open('images.txt').read().splitlines()


''' numeric labelling of food classes '''

for name in labels:
	#print '\nname',name
	for i,food in enumerate(food_list):
		if food in name:
			#print '\nfood',food
			output_label_list.append(i)

print '\noutput_label_list', len(output_label_list)

output_label = output_label_list[:9000]
input_vectors = vectors_list[:9000]

test_inputs = vectors_list[9000:10000]
test_ground_truth = output_label_list[9000:10000]

#print len(input_label_list)
#print len(l1)
print 'test_inputs', len(test_inputs)
print 'test ground truth', len(test_ground_truth)

output_label = np.asarray(output_label)
test_inputs = np.asarray(test_inputs)


output_label_list = []

y = np.float32(output_label)

for j in output_label:
	k = np.float32(j)
	output_label_list.append(k)
vlist = []
for v in input_vectors:
	l = np.float32(v)
	vlist.append(l)

x = np.asarray(vlist)
y = np.asarray(output_label_list)


svm_params = dict( kernel_type = cv2.SVM_LINEAR,
                    svm_type = cv2.SVM_C_SVC,
                    C=2.67, gamma=5.383 )

''' checking shapes of input and output vectors'''
print 'x shape', np.shape(x)
print 'y shape',np.shape(y)


''' Training using svm algorithm using opencv library '''

svm = cv2.SVM()
svm.train(x,y, params=svm_params)

''' Testing '''
test_result_list = []
for t in range(len(test_inputs)): 
	test = np.float32(test_inputs[t])
	test_result = svm.predict(test)
	test_result_list.append(test_result)

'''evaluating metrics'''
accuracy = accuracy_score(test_ground_truth, test_result_list)
f1_score = f1_score(test_ground_truth,test_result_list, average='macro')
cmatrix =  confusion_matrix(test_ground_truth, test_result_list)

print '\n confusion matrix',cmatrix
print 'accuracy',accuracy
print 'f1 score',f1_score

