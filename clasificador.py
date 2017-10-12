import sys
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from random import randint
import glob
import time
training = np.array(1)
tests = []
firstAcc = 1

def initTrainSet(classifier, matrix):
	global training
	global tests
	Xs, Ds = obtainXFx(matrix)
	maxAcc = 0
	bestX_train = bestD_train = bestX_test = bestD_test = np.array(1)
	for i in range(1):
		X_train, X_test, d_train, d_test = split(Xs,Ds,0.3)
		acc = classifierAccuracy1(classifier,X_train,d_train,X_test,d_test)
		if(maxAcc<=acc):
			maxAcc = acc
			bestX_train = X_train
			bestD_train = d_train
			bestX_test = X_test
			bestD_test = d_test
	#print(bestX_train.ndim, bestD_train.ndim)
	bestX_train.astype('str')
	bestD_train.astype('str')
	bestX_test.astype('str')
	bestD_test.astype('str')
	training = np.append(bestX_train,bestD_train[...,None] ,1)
	tests = [np.append(bestX_train,bestD_train[...,None] ,1)]

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
	def __init__(self, p = set(), acc = 0, fo = 0):
		self.positions = p   	# Lista de enteros
		self.accuracy  = acc	# Float
		self.fo = 0

	def createSubmatrix(self, A):
		# A -> A'
		B = A
		lista = list(self.positions)
		lista.sort()
		lista.reverse()
		for i in lista:
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

def classifierAccuracy1(classifier, trainingData, trainingRespones, testData,
					  testResponses):
	classifier.fit(trainingData,trainingRespones)
	y_pred = classifier.predict(testData)
	return metrics.accuracy_score(testResponses,y_pred)

def classifierAccuracy(classifier, trainingData, testData):
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

def objectiveFunction(s,classifier, alfa=0.85):
	global firstAcc
	global training
	global tests
	s.accuracy = classifierAccuracy(classifier,s.createSubmatrix(training),tests)
	deltaAcc = -(firstAcc - s.accuracy)
	deltaInst = (len(s.positions))/training.shape[0]
	#print(deltaInst)
	s.fo = deltaAcc*alfa + deltaInst*(1-alfa)

def firstBetter(s, Ns, classifier, maxTries=100):
	for n in Ns:
		objectiveFunction(n, classifier)
		#print(n.accuracy, n.fo)
		if(s.fo <= n.fo):
			return n
	return s

def percentageBetter(s , Ns, classifier, percentage=0.2, include=True):
	maxAcc = 0
	bestSol = Solution()
	if include:
		bestSol = s

	porcion = int(len(Ns)*percentage)
	for i in range(porcion):
		objectiveFunction(Ns[i], classifier)
		#print(Ns[i].fo)
		if(bestSol.fo <= Ns[i].fo):
			bestSol = Ns[i]
	return bestSol

def randomNeighbour(s, Ns, classifier):
	i = randint(0,len(Ns)-1)
	objectiveFunction(Ns[i], classifier)
	return (Ns[i])

def neighbours1(s,k=1):
	global training
	result = []
	for i in range(training.shape[0]):
		if not i in s.positions:
			sol = Solution(p=s.positions | {i})
			result.append(sol)
	return result

def localSearch(mejoramiento, instance):
	global training
	global tests
	global firstAcc
	data = txtToMatrix(instance)	
	clf = LinearSVC()
	initTrainSet(clf, data)
	#initTestSet(training)
	s = Solution()
	objectiveFunction(s, clf)
	#print("Fo ini: ",s.fo)
	#print("Fo INI: ", s.fo)
	firstAcc = s.accuracy
	star = s
	#print("#Initial Samples:",training.shape[0] - len(s.positions), "Acc: ", s.accuracy)
	ite = 0
	while True:
		Ns = neighbours1(s)
		prevS = s
		s = mejoramiento(s,Ns,clf)
		if(s == prevS):
			break
		else:
			prevS = s
			#print("Improved:",s.accuracy,"Fo:",s.fo)
		ite +=1
		#print(ite)
	#print("#Iteraciones:",ite)
	#print("#Final Samples:",training.shape[0] - len(s.positions), "Acc:", s.accuracy)
	return len(s.positions) , -firstAcc + s.accuracy, ite

if __name__ == '__main__':
	#localSearch(firstBetter)
	instanceFolder = "datasets"
	instances = glob.glob(instanceFolder + "/*.txt")
	for instance in instances:
		f = open("result_" + instance.split('/')[1], 'w+')
		print("DATASET: " + instance)
		print("percentageBetter")
		print("d_inst , d_acc, total_time, ite")

		f.write("percentageBetter\n")
		for i in range(0,10):
			start_time = time.time()
			d_inst, d_acc, ite = localSearch(percentageBetter, instance)
			total_time = time.time() - start_time
			print(str(d_inst)+","+ str(d_acc) +","+ str(total_time) + "," + str(ite))
			f.write(str(d_inst)+","+ str(d_acc) +","+ str(total_time) + "," +str(ite)+"\n")
		print("firstBetter")
		print("d_inst , d_acc, total_time, ite")
		f.write("\nfirstBetter\n")
		for i in range(0,10):
			start_time = time.time()
			d_inst, d_acc, ite  = localSearch(firstBetter, instance)
			total_time = time.time() - start_time
			print(str(d_inst)+","+ str(d_acc) +","+ str(total_time) + "," + str(ite))
			f.write(str(d_inst)+","+ str(d_acc) +","+ str(total_time) + "," +str(ite)+"\n")
			
		f.close()




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