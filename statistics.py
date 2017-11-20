import glob
import numpy as np

if __name__ == '__main__':
	# Para cada tamanio de instancia
	# localSearch(firstBetter, sys.argv[1])
	resultsFolder = "Results/"
	sizeFolders = ["Large/"]#["Small/", "Medium/"] #Agregar Large cuando termine
	for fd in sizeFolders:
		instanceResultFolder = resultsFolder + fd
		results = glob.glob(instanceResultFolder + "/*.txt")
		for result in results:
			aux = result.split("/")
			stats = resultsFolder + fd + "stats_" + aux[len(aux)-1]
			direcc = resultsFolder + fd + aux[len(aux)-1]
			#my_file = Path(direcc)
			#if my_file.is_file():
			#	print("Ya existe: " + direcc)
			#	continue
			st = open(stats, 'w+')
			f = open(direcc, 'r')
			matrix = []
			corrida = False
			print("Analzing: " + result)
			while True:
				line = f.readline()
				#print(len(line))
				if line == "" and not corrida:
					break
				line.strip('\n')
				data = line.split()
				if(len(data)==1):
					corrida = True
					st.write(data[0] +"\n")
					renglones = ["MediaIni","MediaIS", "MAccIni+-stdev",
								"MAccIS+-stdev", "Media Ite", "Tiempo Medio"]
					string = '\t'.join(x for x in renglones)
					st.write(string +"\n")
					continue
				elif len(data)==0:
					results_data = np.array(matrix,dtype = 'float64')
					#print(results_data)
					mediaIni = np.mean(results_data[:,0])
					mediaIS = np.mean(results_data[:,1])
					mAccIni = np.mean(results_data[:,2])
					mAccIniStDev = np.std(results_data[:,2],ddof = 1) 
					mAccIS = np.mean(results_data[:,3])
					mAccISStDev = np.std(results_data[:,3], ddof = 1)
					mIte = np.mean(results_data[:,4])
					mTime = np.mean(results_data[:,5])
					values = [mediaIni,mediaIS,mAccIni,mAccIS,mIte,mTime]
					values = [x for x in map(str, values)]
					values[2] = str(mAccIni)+"+-"+str(mAccIniStDev)
					values[3] = str(mAccIS)+"+-"+str(mAccISStDev)
					string = '\t'.join(x for x in values)
					st.write(string+'\n')
					matrix = []
					corrida = False
					continue
				matrix.append(data)

			f.close()
			st.close()
