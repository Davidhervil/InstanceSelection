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
	return np.array(matrix)

def obtainXFx(matrix):
	imp = preprocessing.Imputer(missing_values ='NaN')
	X = np.array(matrix[:,:matrix.shape[1]-1],dtype='float64')
	X = imp.fit_transform(X)
	d = matrix[:,matrix.shape[1]-1]
	return X,d

def split(X,d,validate_size,random_stat=randint(11,100)):
	#print("Semilla", random_stat)
	return train_test_split(X,
							d,
							test_size=validate_size,
							random_state=None)

def objectiveFunction(classifier, trainingData, trainingRespones, testData,
					  testResponses):
	classifier.fit(trainingData,trainingRespones)
	y_pred = classifier.predict(testData)
	return metrics.accuracy_score(testResponses,y_pred)

"""def main():
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
"""

def firstBetter(acc,trainingSet,testSet,classifier,maxTries=100):
	(x,d) = trainingSet
	(tx,td) = testSet
	
	bestAcc = acc
	actual = dactual = np.array(1)
	ite = 0
	while ite < maxTries and ite<x.shape[0]:
		actual = np.delete(x,ite,0)
		dactual = np.delete(d,ite,0)
		a = objectiveFunction(classifier,actual,dactual,tx,td)
		if(bestAcc<a):
			return (actual,dactual)
		ite+=1
	return trainingSet

def percentageBetter(acc,trainingSet,testSet,classifier,percentage=0.2, include=True):
	(x,d) = trainingSet
	(tx,td) = testSet
	if include:
		maxAcc = acc
	else:
		maxAcc = 0
	found = False
	rows = int(x.shape[0]*percentage)
	bestX_train = bestX_test = bestD_train = bestD_test = np.array(1)
	for i in range(rows):
		actual = np.delete(x,i,0)
		dactual = np.delete(d,i,0)
		a = objectiveFunction(classifier,actual,dactual,tx,td)
		if(maxAcc<=a):
			maxAcc = acc
			bestX_train = actual
			bestD_train = dactual
			found = True
	if found:
		return (bestX_train,bestD_train)
	else:
		return trainingSet

def randomNeighbour(acc,trainingSet,testSet,classifier):
	(x,d) = trainingSet
	(tx,td) = testSet
	nigger = randint(0,x.shape[0])
	return (np.delete(x,nigger,0),np.delete(d,nigger,0))

def findSolInit(classifier,Xs,Ds):
	maxAcc = 0
	bestX_train = bestX_test = bestD_train = bestD_test = np.array(1)
	for i in range(10):
		X_train, X_test, d_train, d_test = split(Xs,Ds,0.2)
		acc = objectiveFunction(classifier,X_train,d_train,X_test,d_test)
		#print("Hola",acc)
		if(maxAcc<=acc):
			maxAcc = acc
			bestX_train = X_train
			bestX_test = X_test
			bestD_train = d_train
			bestD_test = d_test
	return bestX_train, bestX_test, bestD_train, bestD_test

def localSearch(mejoramiento):
	data = txtToMatrix(sys.argv[1])
	Xs, Ds = obtainXFx(data)
	
	clf = LinearSVC()
	bestX_train, bestX_test, bestD_train, bestD_test = findSolInit(clf,Xs,Ds)

	bestAcc = objectiveFunction(clf,bestX_train,bestD_train,bestX_test,bestD_test)
	eps = 0.001
	actualSol = (bestX_train, bestD_train)
	print("Empezamos: ",actualSol[0].shape[0])
	print("con ",bestAcc)
	while True:
		accS = objectiveFunction(clf,actualSol[0],actualSol[1],bestX_test,bestD_test)
		s = mejoramiento(accS,actualSol,(bestX_test,bestD_test),clf)
		if(s == actualSol):
			break
		else:
			actualSol = s
	print("Quedamos: ",actualSol[0].shape[0])
	print("con ",accS)

def main():
	data = txtToMatrix(sys.argv[1])
	Xs, Ds = obtainXFx(data)
	

	clf = LinearSVC()
	maxAcc = 0
	bestX_train = bestX_test = bestD_train = bestD_test = np.array(1)
	for i in range(10):
		X_train, X_test, d_train, d_test = split(Xs,Ds,0.2)
		acc = objectiveFunction(clf,X_train,d_train,X_test,d_test)
		print("Hola",acc)
		if(maxAcc<acc):
			maxAcc = acc
			bestX_train = X_train
			bestX_test = X_test
			bestD_train = d_train
			bestD_test = d_test
	print(maxAcc)

if __name__ == '__main__':
	epsilon = 0.05
	localSearch(percentageBetter)