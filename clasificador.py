import sys
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from random import randint

def txtToMatrix(path):
	f = open(path,'r')
	matrix = []
	for line in f:
		line = line.strip('\n')
		data = line.split(',')
		matrix.append(data)
	return matrix

def obtainXD(matrix):
	npmat = np.array(matrix)
	X = np.array(npmat[:,:npmat.shape[1]-1],dtype='float64')
	d = npmat[:,npmat.shape[1]-1]
	return X,d

def split(X,d,validate_size,random_stat=randint(11,100)):
	#print("Semilla", random_stat)
	return train_test_split(X,
							d,
							test_size=validate_size,
							random_state=None)

def main():
	data = txtToMatrix(sys.argv[1])
	X, d = obtainXD(data)
	X_train, X_test, d_train, d_test = split(X,d,0.2)
	#X_train = preprocessing.scale(X_train)
	#X_test = preprocessing.scale(X_test)
	#X_train = preprocessing.normalize(X_train)
	#X_test = preprocessing.normalize(X_test)
	clf = SGDClassifier(n_jobs = -1,shuffle = False)
	clf.fit(X_train, d_train)
	y_pred = clf.predict(X_test)
	#print(y_pred)
	print (metrics.accuracy_score(d_test, y_pred))
	#print (metrics.classification_report(d_test, y_pred))
	#print(clf.coef_)
	print(clf.n_iter_)
	print("###")
	clf2 = LinearSVC()
	clf2.fit(X_train, d_train)
	y_pred = clf2.predict(X_test)
	#print(y_pred)
	print (metrics.accuracy_score(d_test, y_pred))
	#print (metrics.classification_report(d_test, y_pred))
	#print(clf.coef_)

if __name__ == '__main__':
	main()