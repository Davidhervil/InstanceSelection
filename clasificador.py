import sys
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn import preprocessing
from sklearn import metrics
from random import randint, random, sample, shuffle
import glob
import time
from pathlib import Path
from math import exp, log
from multiprocessing import Pool

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
	def __init__(self, p, acc = 0, fo = 0):
		self.positions = p   	# Lista de enteros
		self.accuracy  = acc	# Float
		self.fo = 0

	def createSubmatrix(self, A):
		# A -> A'
		B = A
		lista = list(self.positions)
		lista.sort()
		lista.reverse()
		for i in range(len(self.positions)-1,-1,-1):
			if self.positions[i] == 1:
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
	try:
		sss = StratifiedShuffleSplit(n_splits=1, test_size=validate_size)
		for train_index, test_index in sss.split(X, d):
			#print("TRAIN:", train_index, "TEST:", test_index)
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = d[train_index], d[test_index]
		return X_train, X_test, y_train, y_test
	except:
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
	deltaInst = (sum(s.positions))/training.shape[0]
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
	bestSol = Solution([0]*training.shape[0])
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
		if s.positions[i]==0:
			new = s.positions.copy()
			new[i] = 1
			sol = Solution(p=new.copy())
			result.append(sol)
	return result

def k_neighbours(s,k=2):
	global training
	res = neighbours1(s)
	result = []
	for i in range(training.shape[0]):
		extract = s.positions.copy()
		counter = 0
		for j in range(i,training.shape[0]):
			if s.positions[j]==0:
				extract[j] = 1
				counter += 1
				if counter >= k:
					sol = Solution(p=extract.copy())
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
	s = Solution([0]*training.shape[0])					# Inicializar Solucion.
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
	s = Solution([0]*training.shape[0])					# Inicializar Solucion.
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
			try:
				sR = randomNeighbour(s, Ns, clf)
			except:
				k = k + 1
				continue
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

	result = [training.shape[0], training.shape[0] - sum(s.positions) , firstAcc, s.accuracy, ite]
	string = '\t'.join(str(x) for x in result)
	return string

def RVNS(vecindades, mejoramiento, instance):
	global training
	global tests
	global firstAcc
	data = txtToMatrix(instance)	# Transformar archivo de texto a matriz
	clf = LinearSVC()				# Inicializar clasificador a utilizar.
	initTrainSet(clf, data)			# Inicializar conjunto de ENTRENAMIENTO y PRUEBAS.
	s = Solution([0]*training.shape[0])					# Inicializar Solucion.
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
			try:
				sP = randomNeighbour(s, Ns, clf)
			except:
				k = k + 1
				continue
			if (sP.fo <= s.fo):		# Condicion de cambio de vecindad.
				k = k + 1
			else:
				s = sP
				k = 0
		if ite > 1000:
			break

		ite +=1
	
	# FORMATO: d_inst  | first_acc | final_acc | #iter | time

	result = [training.shape[0], training.shape[0] - sum(s.positions) , firstAcc, s.accuracy, ite]
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
	s = Solution([0]*training.shape[0])					# Inicializar Solucion.
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
			try:
				sR = randomNeighbour(s, Ns, clf)
			except:
				k = k + 1
				continue
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

	result = [training.shape[0], training.shape[0] - sum(estrella.positions) , firstAcc, estrella.accuracy, ite]
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

def simulatedAnnealing(Tmax, vecindad, g, instance, s = None, clf = None):
	#Mejores Tmax: 23, 60, 1000
	# Tarda menos con kneighbours
	global training
	global tests
	global firstAcc
	aux = False
	if s == None:	
		data = txtToMatrix(instance)	# Transformar archivo de texto a matriz
		clf = LinearSVC()				# Inicializar clasificador a utilizar.
		initTrainSet(clf, data)			# Inicializar conjunto de ENTRENAMIENTO y PRUEBAS.
		s = Solution([0]*training.shape[0])					# Inicializar Solucion.
		objectiveFunction(s, clf)		# Calcular la funcion objetivo de la solucion inicial.
		firstAcc = s.accuracy 			# Primer valor de Fo para usar de referencia.
		aux = True
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
		if neighbourhood == 0:
			break
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

	if aux == True:
		result = [training.shape[0], training.shape[0] - sum(leBest.positions) , firstAcc, leBest.accuracy, ite]
		string = '\t'.join(str(x) for x in result)
		return string
	return leBest

def genRandSol(many):
	result = []
	for k in range(many):
		numero = randint(0,int(training.shape[0]*0.4))
		aux = [1 for i in range(numero)]+[0]*(training.shape[0]-numero)
		shuffle(aux)
		result.append(Solution(p=aux.copy()))

	return result

def wrapperF(x):
		sol = x[0]
		clf = x[1]
		return simulatedAnnealing(23,neighbours1,g_alfa,None,sol,clf)

def beeIntensify(sites,bees,clf):
	

	BestIntensified = []
	for site in sites :
		Ns = k_neighbours(site) + [site]
		sampled = sample(Ns,min(bees,len(Ns)))
		sampled = [(s,clf) for s in sampled]
		sampleds = []
		with Pool(bees) as p:
			sampleds = p.map(wrapperF,sampled)
		if sampleds == []:
			continue
		BestIntensified.append(max(sampleds, key = lambda x: x.fo))

	return BestIntensified

def bee(n, m, e, elite, other, instance):
	global training
	global tests
	global firstAcc
	data = txtToMatrix(instance)	# Transformar archivo de texto a matriz
	clf = LinearSVC()				# Inicializar clasificador a utilizar.
	initTrainSet(clf, data)			# Inicializar conjunto de ENTRENAMIENTO y PRUEBAS.
	s = Solution([0]*training.shape[0])					# Inicializar Solucion.
	objectiveFunction(s, clf)		# Calcular la funcion objetivo de la solucion inicial.
	firstAcc = s.accuracy 			# Primer valor de Fo para usar de referencia.

	P = genRandSol(n)
	for sol in P:
			objectiveFunction(sol,clf)
	Pm = []
	ite = 0
	best = s
	while ite<5:	
		for sol in P:
			if len(Pm) < m:
				Pm.append(sol)
			else:
				mini = min(Pm, key = lambda x: x.fo)
				if sol.fo > mini.fo:
					Pm.remove(mini)
					Pm.append(sol)

		Pm = sorted(Pm, key= lambda x : x.fo)	

		bestEs = beeIntensify(Pm[m-e:m], elite, clf)
		bestOther = beeIntensify(Pm[:m-e], other, clf)

		Pm = bestEs + bestOther	


		newOnes = genRandSol(n-m)
		for new in newOnes:
			objectiveFunction(new,clf)
		P = Pm + newOnes

		ultimateBee = max(P, key = lambda x : x.fo)
		if ultimateBee.fo > best.fo:
			best = ultimateBee
			ite = 0
		else:
			ite+=1

	return best
# -------------------------------------- CHC ------------------------------- #
def distance(s1, s2):
	result = 0
	for i in range(len(s1.positions)):
		result += s1.positions[i]^s2.positions[i]
	return result

def findBestMatch(s1, poblac):
	maxim = 0
	result = s1
	for p in poblac:
		act = distance(s1,p)
		if act>=maxim:
			maxim  = act
			result = p
	return result

def randomCrossover(s1,s2): #tal vez pasa lo mismo que el genRandSol v.1
	result = Solution(p=s1.positions.copy())
	for i in range(len(s1.positions)):
		numero = randint(0,1)
		if numero == 1:
			result.positions[i] = s1.positions[i]
		else:
			result.positions[i] = s2.positions[i]
	return result

def convergence(poblacion, maxim):
	acc = np.array(poblacion[0].positions)
	for i in range(1,len(poblacion)):
		acc += np.array(poblacion[i].positions)
	acc /= len(poblacion).positions

	mapper = np.vectorize(lambda x: int(x))
	centerPos = mapper(acc).tolist() 
	centroid = Solution(p=centerPos)
	
	distances = map(lambda s : distance(s,centroid),poblacion)
	meanDist = sum(distances)/len(distances)*1.0

	if meanDist < maxim:
		return True
	return False

def CHC(n, ite, conv):
	data = txtToMatrix(instance)	# Transformar archivo de texto a matriz
	clf = LinearSVC()				# Inicializar clasificador a utilizar.
	initTrainSet(clf, data)			# Inicializar conjunto de ENTRENAMIENTO y PRUEBAS.
	aux = 0
	best = None
	while(aux < ite):
		while(True):
			poblacion = genRandSol(n)
			for p in poblacion:
				objectiveFunction(p, clf)				# Calcular la funcion objetivo de la solucion inicial.
				oposite = findBestMatch(p,poblacion)	# Encontrar el polo opuesto, porque atrae (?)
				# Calcular dos hijos para cada par de polos opuestos (Emely buscaba al menos dos).
				son 	 = randomCrossover(p, oposite)	
				daughter = randomCrossover(p, oposite)
				objectiveFunction(son, clf)				# Evaluar las nuevas soluciones.
				objectiveFunction(daughter, clf)
				poblacion.append(son)					# Agregar las nuevas soluciones.
				poblacion.append(daughter)

			# Mantener los n mejores individuos (Conservative Selection Strategy).
			poblacion = sorted(poblacion, key= lambda x : x.fo)	
			poblacion = poblacion[:n] # OJOJOJOJOJOJO esto puede agregar incesto, pues solo te estas quedando con las
									  # n mejores en lugar de renovar la generacion
			# Reinicio
			if (convergence(poblacion, conv)):
				break
		aux += 1
		if(best == None or best.fo<poblacion[0].fo):
			best = poblacion[0]
	return best

if __name__ == '__main__':

	# Para cada tamanio de instancia
	# localSearch(firstBetter, sys.argv[1])
	datasetFolder = "datasets/"
	resultsFolder = "Results/"
	sizeFolders = ["Small/", "Medium/", "Large/"]	# Tamanos de problemas
	mejoramientos = [firstBetter, percentageBetter]	# Tipos de mejoramiento (Busqueda Local)
	vecindades=[neighbours1, k_neighbours, lambda s : k_neighbours(s, k=3), lambda s : k_neighbours(s, k=4)]						# Tipos de vecindades

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
			#f = open(direcc, 'w+')
			print("Running: " + instance)
			
			#f.write("\nRVNS\n")
			for i in range(0,10):
				start_time = time.time()
				result = "ERROR"
				if sys.argv[1] == "VNS":
					result = VNS(vecindades, firstBetter, instance)
				elif sys.argv[1] == "SA":
					result = simulatedAnnealing(23,vecindades[1],g_alfa,instance)
				elif sys.argv[1] == "SVNS":
					result = SVNS(vecindades, firstBetter, instance)
				elif sys.argv[1] == "RVNS":
					result = RVNS(vecindades, firstBetter, instance)
				elif sys.argv[1] == "BA":
					result = bee(n=7, m=5, e=2, elite=2, other=1, instance=instance)
				elif sys.argv[1] == "CHC":
					result = CHC(n=10, ite=5, conv=100)
				else:
					print(sys.argv[1]," Opcion invalida.")
				total_time = time.time() - start_time
				result = [training.shape[0], training.shape[0] - sum(result.positions) , firstAcc, result.accuracy]
				result = '\t'.join(str(x) for x in result)
				print(str(result) + "\t" + str(total_time))
				#f.write(str(result) + "\t" + str(total_time) + "\n")

			print("Results: " + direcc+ "\n")
			#f.close()
