import glob
import time
import numpy as np

def txtToMatrix(path):
	# Crea la matriz y la devuelve como arreglo de numpy
	f = open(path,'r')
	matrix = []
	for line in f:
		line = line.strip('\n')
		data = line.split(',')
		matrix.append(data)
	return np.array(matrix)

instanceFolder = "datasets"
instances = glob.glob(instanceFolder + "/*.txt")
for instance in instances:
	matrix = txtToMatrix(instance)
	print(instance,matrix.shape[0]*matrix.shape[1])