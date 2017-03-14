import math
#from sklearn import svm
import numpy as np
import cv2
from string import digits
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import re

def unison_shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


food_list = open('class_labels.txt').read().splitlines()


'''10000 image vectors'''

lines = open('output_alldata.txt').read().splitlines()

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
print 'total number of vectors',len(vectors)


'''input labels'''
labels = []
output_label_list = []

labels = open('100k_images.txt').read().splitlines()
print 'total number of images',len(	labels )



''' numeric labelling of food classes '''

labels = np.asarray(labels)
vectors = np.asarray(vectors)
#labels,vectors = unison_shuffle(labels,vectors)

for name in labels:
	#print '\nname',name
	s = name.split('/')[-1]
	s = s.split('.')[0]
	res = s.translate(None, digits)
	for i,food in enumerate(food_list):
		if food == res:
			#print '\nfood',food,'res', res
			output_label_list.append(i)

print '\noutput_label_list', len(output_label_list)

output_label = output_label_list[:90000]
input_vectors = vectors[:90000]

test_inputs = vectors[90000:101000]
test_ground_truth = output_label_list[90000:101000]

#print len(input_label_list)
#print len(l1)
print 'test_inputs', len(test_inputs)
#print 'input test_ground_truth ',test_ground_truth
print 'test ground truth', len(test_ground_truth)

output_label = np.asarray(output_label)
test_inputs = np.asarray(test_inputs)

olabel_list  = []

y = np.float32(output_label)

for j in output_label:
	k = np.float32(j)
	olabel_list.append(k)
vlist = []
for v in input_vectors:
	l = np.float32(v)
	vlist.append(l)

x = np.asarray(vlist)
y = np.asarray(olabel_list)


#svm_params = dict( kernel_type = cv2.SVM_LINEAR,
#                    svm_type = cv2.SVM_C_SVC,
#                    C=2.67, gamma=5.383 )

svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setP(0.2)
svm.setType(cv2.ml.SVM_EPS_SVR)
#svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 1.e-06))
svm.setC(12.5)
svm.setGamma(5.383)


''' checking shapes of input and output vectors'''
print 'x shape', np.shape(x)
print 'y shape',np.shape(y)


''' Training using svm algorithm using opencv library '''


#svm.train(x,y, params=svm_params)
svm.train(x, cv2.ml.ROW_SAMPLE, y)


''' Testing '''
#test_result_list = []
#for t in range(len(test_inputs)): 
#	#test = np.float32(test_inputs[t])
#	test_result = svm.predict(test)
#	test_result_list.append(test_result)

test_data = np.array(test_inputs, np.float32)
test_result_list = svm.predict(test_data)
print 'input data',test_inputs
#print 'test_result', len(test_result_list)
#print 'test_result', len(test_result_list[1].tolist())

test_res = test_result_list[1].tolist()
# print 'test res len',len(test_res)
# print 'test res', test_res
# print '-----------------'
# print 'test ground truth', test_ground_truth

final = []
for i in test_res:
	final.append(int(i[0]))

print 'final',final

#print 'checking output'
#for i,j in zip(final,test_ground_truth):
#	print 'prediction:', i, '\t','ground truth:',j

'''evaluating metrics'''
accuracy = accuracy_score(test_ground_truth, final)
f1_score = f1_score(test_ground_truth,final, average='macro')
cmatrix =  confusion_matrix(test_ground_truth,final)

print '\n confusion matrix',cmatrix
print 'accuracy',accuracy
print 'f1 score',f1_score

