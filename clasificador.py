import sys
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from random import randint

training = np.array(1)
tests = []

def initTrainSet(classifier, matrix):
	global training
	Xs, Ds = obtainXFx(matrix)
	maxAcc = 0
	bestX_train = bestD_train = np.array(1)
	for i in range(10):
		X_train, X_test, d_train, d_test = split(Xs,Ds,0.2)
		acc = objectiveFunction1(classifier,X_train,d_train,X_test,d_test)
		if(maxAcc<=acc):
			maxAcc = acc
			bestX_train = X_train
			bestD_train = d_train
	#print(bestX_train.ndim, bestD_train.ndim)
	bestX_train.astype('str')
	bestD_train.astype('str')
	training = np.append(bestX_train,bestD_train[...,None] ,1)

def initTestSet(matrix, k=10):
	global tests
	filas = matrix.shape[0]
	quantity = int(filas / k)
	for i in range(0, filas, quantity):
		if(i+quantity < filas):
			m = matrix[i:(i+quantity),:]
		else:
			m = matrix[i:,:]
		tests.append(m)

class Solution:
	def __init__(self, p = set(), acc = 0):
		self.positions = p   	# Lista de enteros
		self.accuracy  = acc	# Float

	def createSubmatrix(self, A):
		# A -> A'
		B = A
		for i in self.positions:
			B = np.delete(B, i, 0)
		return B

#############################################################################
def txtToMatrix(path):
	# Crea la matriz y la devuelve como arreglo de numpy
	f = open(path,'r')
	matrix = []
	for line in f:
		line = line.strip('\n')
		data = line.split(',')
		matrix.append(data)
	return np.array(matrix)

def obtainXFx(matrix):
	# Separa las variables de los resultados.
	imp = preprocessing.Imputer(missing_values ='NaN')
	X = np.array(matrix[:,:matrix.shape[1]-1],dtype='float64')
	X = imp.fit_transform(X)
	d = matrix[0:matrix.shape[0],matrix.shape[1]-1]
	return X, d

def split(X,d,validate_size,random_stat=randint(11,100)):
	#print("Semilla", random_stat)
	return train_test_split(X,
							d,
							test_size=validate_size,
							random_state=None)

def objectiveFunction1(classifier, trainingData, trainingRespones, testData,
					  testResponses):
	classifier.fit(trainingData,trainingRespones)
	y_pred = classifier.predict(testData)
	return metrics.accuracy_score(testResponses,y_pred)

def objectiveFunction(classifier, trainingData, testData):
	# Funcion que recibe un clasificador, una matriz de entrenamiento y
	# n matrices de prueba y retorna el valor promedio de la precision.
	x_train, d_train = obtainXFx(trainingData)
	classifier.fit(x_train, d_train)
	acum = 0
	for test in testData:
		x_test, d_test = obtainXFx(test)
		y_pred = classifier.predict(x_test)
		acum += metrics.accuracy_score(d_test,y_pred)
	return float(acum/len(testData))

def firstBetter(s, Ns, classifier, maxTries=100):
	global training
	global tests
	for n in Ns:
		n.accuracy = objectiveFunction(classifier, n.createSubmatrix(training), tests)
		if(s.accuracy <= n.accuracy):
			return n
	return s

def percentageBetter(s , Ns, classifier, percentage=0.2, include=True):
	global training
	global tests
	maxAcc = 0
	bestSol = Solution()
	if include:
		bestSol = s

	porcion = int(len(Ns)*percentage)
	for i in range(porcion):
		Ns[i].accuracy = objectiveFunction(classifier,Ns[i].createSubmatrix(training),tests)
		if(bestSol.accuracy <= Ns[i].accuracy):
			bestSol = Ns[i]
	return bestSol

def randomNeighbour(s, Ns, classifier):
	global training
	global tests
	i = randint(0,len(Ns)-1)
	Ns[i].accuracy = objectiveFunction(classifier,Ns[i].createSubmatrix(training),tests)
	return (Ns[i])

def neighbours1(s,k=1):
	global training
	global tests
	result = []
	for i in range(training.shape[0]):
		if not i in s.positions:
			s = Solution(p=s.positions | {i})
			result.append(s)
	return result

def localSearch(mejoramiento):
	global training
	global tests
	data = txtToMatrix(sys.argv[1])	
	clf = LinearSVC()
	initTrainSet(clf, data)
	initTestSet(training)
	s = Solution()
	s.accuracy = objectiveFunction(clf, s.createSubmatrix(training), tests)
	star = s
	print("INI#Samples: ",training.shape[0] - len(s.positions), "Acc: ", s.accuracy)
	while True:
		Ns = neighbours1(s)
		prevS = s
		s = mejoramiento(s,Ns,clf)
		if(s == prevS):
			break
		else:
			prevS = s
	print("#Samples: ",training.shape[0] - len(s.positions), "Acc: ", s.accuracy)

def main():
	data = txtToMatrix(sys.argv[1])
	Xs, Ds = obtainXFx(data)
	

	clf = LinearSVC()
	maxAcc = 0
	bestX_train = bestX_test = bestD_train = bestD_test = np.array(1)
	for i in range(10):
		X_train, X_test, d_train, d_test = split(Xs,Ds,0.1)
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