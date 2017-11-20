import sys
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn import preprocessing
from sklearn import metrics
from random import randint, random
import glob
import time
from pathlib import Path
from math import exp, log

# GLOBAL VARIABLES
training = np.array(1)	# Conjunto de entrenamiento
tests = []				# Conjunto de pruebas
firstAcc = 1			# Primera precision

#------------------------- PREPROCESAMIENTO DE DATOS ------------------------#
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
	tests = [np.append(bestX_test,bestD_test[...,None] ,1)]

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

def objectiveFunction(s,classifier, alfa=0.8):
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

def k_neighbours(s,k=2):
	global training
	res = neighbours1(s)
	result = []
	for i in range(training.shape[0]):
		extract = set()
		for j in range(i,training.shape[0]):
			if not j in s.positions:
				extract.add(j)
				if len(extract) >= k:
					sol = Solution(p=s.positions | extract)
					result.append(sol)
					break
	return result
# M vecinos mas cercanos.
# K_neighbours.
# Sacar de las clases mas grandotas.

#------------------------- BUSQUEDA LOCAL -------------------------#
def setupExperiment(instance):
	global training
	global tests
	global firstAcc
	global clf
	training = np.array(1)	# Conjunto de entrenamiento
	tests = []				# Conjunto de pruebas
	firstAcc = 1			# Primera precision
	clf = None
	data = txtToMatrix(instance)	# Transformar archivo de texto a matriz
	clf = LinearSVC()				# Inicializar clasificador a utilizar.
	initTrainSet(clf, data)			# Inicializar conjunto de ENTRENAMIENTO y PRUEBAS.
	s = Solution()					# Inicializar Solucion.
	objectiveFunction(s, clf)		# Calcular la funcion objetivo de la solucion inicial.
	firstAcc = s.accuracy 			# Primer valor de Fo para usar de referencia.
	ite = 0		
	return s

def localSearch(vecindad, mejoramiento, s, clf):
	""" 
		- Vecindad: Tipo de vecindad a utilizar.
		- Mejoramiento: Tipo de mejoramiento a utilizar.
		- s: Solucion actual
	"""
	#setupExperiment()
	ite = 0
	while True:			
		Ns = vecindad(s)			# Obtener la vecindad de s.
		prevS = s 					# COMP: Verificar mejoramiento.
		try:
			s = mejoramiento(s,Ns,clf)	# Realizar mejoramiento.
		except:
			break
		if(s == prevS):				# Condicion de parada: No mejoramiento.
			break
		ite +=1
	return s

	# FORMATO: d_inst  | first_acc | final_acc | #iter | time
	#result = [training.shape[0], training.shape[0] - len(s.positions) , firstAcc, s.accuracy, ite]
	#string = '\t'.join(str(x) for x in result)
	#return string


#----------------- Metaheuristicas de trayectoria -----------------#
# Se decide utilizar la busqueda local con PRIMER MEJOR.
# 
def VNS(vecindades, mejoramiento, instance):
	global training
	global tests
	global firstAcc
	data = txtToMatrix(instance)	# Transformar archivo de texto a matriz
	clf = LinearSVC()				# Inicializar clasificador a utilizar.
	initTrainSet(clf, data)			# Inicializar conjunto de ENTRENAMIENTO y PRUEBAS.
	s = Solution()					# Inicializar Solucion.
	objectiveFunction(s, clf)		# Calcular la funcion objetivo de la solucion inicial.
	firstAcc = s.accuracy 			# Primer valor de Fo para usar de referencia.
	ite = 0	
	start_time = time.time()		# Tiempo de inicio.
	ite = 0							# INFO: Numero de iteraciones			
	while (True):
		k = 0
		prevS = s
		while(k< len(vecindades)):				# Probar todas las vecindades.
			Ns = vecindades[k](s)				# Generar vecino random
			sR = randomNeighbour(s, Ns, clf)	
			#print("Probando con k: ",k)
			sP = localSearch(vecindades[k], mejoramiento, sR, clf)
			if (sP.fo <= s.fo):		# Condicion de cambio de vecindad.
				k = k + 1
			else:
			#	print("mejoro")
				s = sP
				k = 0
		if s.fo == prevS.fo or ite > 5:
			break

		ite +=1
	
	# FORMATO: d_inst  | first_acc | final_acc | #iter | time

	result = [training.shape[0], training.shape[0] - len(s.positions) , firstAcc, s.accuracy, ite]
	string = '\t'.join(str(x) for x in result)
	return string

def RVNS(vecindades, mejoramiento, instance):
	global training
	global tests
	global firstAcc
	data = txtToMatrix(instance)	# Transformar archivo de texto a matriz
	clf = LinearSVC()				# Inicializar clasificador a utilizar.
	initTrainSet(clf, data)			# Inicializar conjunto de ENTRENAMIENTO y PRUEBAS.
	s = Solution()					# Inicializar Solucion.
	objectiveFunction(s, clf)		# Calcular la funcion objetivo de la solucion inicial.
	firstAcc = s.accuracy 			# Primer valor de Fo para usar de referencia.
	ite = 0	
	start_time = time.time()		# Tiempo de inicio.
	ite = 0							# INFO: Numero de iteraciones			
	while (True):
		k = 0
		prevS = s
		while(k< len(vecindades)):				# Probar todas las vecindades.
			Ns = vecindades[k](s)				# Generar vecino random
			sP = randomNeighbour(s, Ns, clf)	
			if (sP.fo <= s.fo):		# Condicion de cambio de vecindad.
				k = k + 1
			else:
				s = sP
				k = 0
		if s.fo == prevS.fo or ite > 5:
			break

		ite +=1
	
	# FORMATO: d_inst  | first_acc | final_acc | #iter | time

	result = [training.shape[0], training.shape[0] - len(s.positions) , firstAcc, s.accuracy, ite]
	string = '\t'.join(str(x) for x in result)
	return string

def proba_SVNS(s, sP, alfa = 0.3):
	return sP.fo + alfa*(abs(sP.fo-s.fo)) > s.fo

def SVNS(vecindades, mejoramiento, instance):
	global training
	global tests
	global firstAcc
	data = txtToMatrix(instance)	# Transformar archivo de texto a matriz
	clf = LinearSVC()				# Inicializar clasificador a utilizar.
	initTrainSet(clf, data)			# Inicializar conjunto de ENTRENAMIENTO y PRUEBAS.
	s = Solution()					# Inicializar Solucion.
	objectiveFunction(s, clf)		# Calcular la funcion objetivo de la solucion inicial.
	firstAcc = s.accuracy 			# Primer valor de Fo para usar de referencia.
	ite = 0	
	start_time = time.time()		# Tiempo de inicio.
	ite = 0							# INFO: Numero de iteraciones
	estrella = s			
	while (True):
		k = 0
		prevS = s
		while(k< len(vecindades)):				# Probar todas las vecindades.
			Ns = vecindades[k](s)				# Generar vecino random
			sR = randomNeighbour(s, Ns, clf)	
			#print("Probando con k: ",k)
			sP = localSearch(vecindades[k], mejoramiento, sR, clf)
			if sP.fo >= s.fo or proba_SVNS(s,sP):
				s = sP
				k = 0
				if s.fo > estrella.fo:
					estrella = s
			else:		# Condicion de cambio de vecindad.
				k = k + 1
		if s.fo == prevS.fo or ite > 5:
			break

		ite +=1
	
	# FORMATO: d_inst  | first_acc | final_acc | #iter | time

	result = [training.shape[0], training.shape[0] - len(estrella.positions) , firstAcc, estrella.accuracy, ite]
	string = '\t'.join(str(x) for x in result)
	return string

def g_kConst(T, kConst, k):
	return T - kConst

def g_alfa(T, alfa = 0.7, k=0):
	#alfa debe estar entre [0.5 , 0.9]
	return alfa*T

def g_log(T0,k): #TURBO LENTO
	return T0/log(k)

def g_ratio(T0,k):
	return T/(1+k)

def simulatedAnnealing(Tmax, vecindad, g, instance):
	#Mejores Tmax: 23, 60, 1000
	# Tarda menos con kneighbours
	global training
	global tests
	global firstAcc
	data = txtToMatrix(instance)	# Transformar archivo de texto a matriz
	clf = LinearSVC()				# Inicializar clasificador a utilizar.
	initTrainSet(clf, data)			# Inicializar conjunto de ENTRENAMIENTO y PRUEBAS.
	s = Solution()					# Inicializar Solucion.
	objectiveFunction(s, clf)		# Calcular la funcion objetivo de la solucion inicial.
	firstAcc = s.accuracy 			# Primer valor de Fo para usar de referencia.
	ite = 0	
	start_time = time.time()		# Tiempo de inicio.
	ite = 0							# INFO: Numero de iteraciones
	leBest = s
	T = Tmax
	noImprovement = 0
	maxStall = 100
	iteracion = 0
	while noImprovement < maxStall:
		improved = False
		rejected = 0
		Ns = vecindad(s)
		neighbourhood = len(Ns)
		while rejected/neighbourhood < 0.6:
			try:
				sP = randomNeighbour(s, Ns, clf)
			except:
				break

			deltaC = sP.fo - s.fo
			if deltaC > 0:
				s = sP
				if s.fo > leBest.fo:
					leBest = s
					improved = True
					#print("Quitados: ",len(s.positions))
				Ns = vecindad(s)
				neighbourhood = len(Ns)
				rejected = 0
			else:
				r = random()
				if r < exp(deltaC/T) :
					s = sP
					Ns = vecindad(s)
					neighbourhood = len(Ns)
					rejected = 0
				else:
					rejected += 1
					#print("Rejected: ",rejected)

		T = g(T, k = iteracion)
		if not improved :
			noImprovement += 1
		ite += 1
		#print(ite)

	result = [training.shape[0], training.shape[0] - len(leBest.positions) , firstAcc, leBest.accuracy, ite]
	string = '\t'.join(str(x) for x in result)
	return string



if __name__ == '__main__':

	# Para cada tamanio de instancia
	# localSearch(firstBetter, sys.argv[1])
	datasetFolder = "datasets/"
	resultsFolder = "Results/"
	sizeFolders = ["Small/", "Medium/", "Large/"]	# Tamanos de problemas
	mejoramientos = [firstBetter, percentageBetter]	# Tipos de mejoramiento (Busqueda Local)
	vecindades=[neighbours1, k_neighbours]						# Tipos de vecindades

	for fd in sizeFolders:
		instanceFolder = datasetFolder + fd 
		instances = glob.glob(instanceFolder + "/*.txt")
		for instance in instances:
			aux = instance.split("/")
			direcc = resultsFolder + fd + "result_" + aux[len(aux)-1]
			my_file = Path(direcc)
			if my_file.is_file():
				print("Ya existe: " + direcc)
				continue
			f = open(direcc, 'w+')
			print("Running: " + instance)
			
			f.write("\nSVNS\n")
			for i in range(0,10):
				start_time = time.time()
				#result = VNS(vecindades, firstBetter, instance)
				#result = simulatedAnnealing(23,vecindades[1],g_alfa,instance)
				result = SVNS(vecindades, firstBetter, instance)
				total_time = time.time() - start_time
				print(str(result) + "\t" + str(total_time))
				f.write(str(result) + "\t" + str(total_time) + "\n")

			print("Results: " + direcc+ "\n")
			f.close()
