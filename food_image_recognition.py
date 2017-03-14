import numpy as np
import cv2
from string import digits
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import re
from sklearn.svm import SVC
from sklearn.externals import joblib


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
    vectors_list.append(vals)


vectors = np.asarray(vectors_list)
print 'total number of vectors',len(vectors)


'''input labels'''
labels = []
output_label_list = []

labels = open('100k_images.txt').read().splitlines()
print 'total number of images',len(	labels )


''' numeric labelling of food classes '''

labels = np.asarray(labels)
labels,vectors = unison_shuffle(labels,vectors)


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
print '\noutput_label_list', output_label_list

output_label = output_label_list[:90000]
input_vectors = vectors[:90000]
output_label = np.asarray(output_label)

test_inputs = vectors[90000:101000]
test_ground_truth = output_label_list[90000:101000]


print 'test_inputs', len(test_inputs)
print 'test ground truth', len(test_ground_truth)


x = input_vectors.astype(np.float32, copy=False)
y = output_label.astype(np.float32, copy=False)

clf = SVC()
clf.fit(x, y)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

filename = 'svm_classifier_alldata.joblib.pkl'
joblib.dump(clf, filename, compress=9)


print 'output label', y
''' checking shapes of input and output vectors'''
print 'x shape', np.shape(x)
print 'y shape',np.shape(y)


''' Testing '''
test_result_list = clf.predict(test_inputs)
print 'test_result',test_result_list

'''evaluating metrics'''
accuracy = accuracy_score(test_ground_truth, test_result_list)
f1_score = f1_score(test_ground_truth,test_result_list, average='macro')
cmatrix =  confusion_matrix(test_ground_truth,test_result_list)

print '\n confusion matrix',cmatrix
print 'accuracy',accuracy
print 'f1 score',f1_score


