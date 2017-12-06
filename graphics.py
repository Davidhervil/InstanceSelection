import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import math

if __name__ == '__main__':
	# Para cada tamanio de instancia
	# localSearch(firstBetter, sys.argv[1])
	resultsFolder = "Results/"
	sizeFolders = ["Small/", "Medium/", "Large/"] #Agregar Large cuando termine
	metaHsfolders = ["GA/","BA/"]#["VNS/","SVNS/","RVNS/","SA/"]
	for meta in  metaHsfolders:
		for fd in sizeFolders:
			instances_names =[] 
			instanceResultFolder = resultsFolder + meta + fd
			results = glob.glob(instanceResultFolder + "/*.txt")
			instances_results_data = []
			for result in results:
				aux = result.split("/")
				stats = resultsFolder + meta + fd + "stats_" + aux[len(aux)-1]
				direcc = resultsFolder + meta + fd + aux[len(aux)-1]
				instance_name = aux[len(aux)-1].split('_')
				instance_name = instance_name[1].replace(".txt","")
				instances_names.append(instance_name)
				#my_file = Path(direcc)
				#if my_file.is_file():
				#	print("Ya existe: " + direcc)
				#	continue
				#st = open(stats, 'w+')
				f = open(direcc, 'r')
				matrix = []
				corrida = False
				print("Analzing: " + result)
				line = f.readline()
				while True: #Esta version asume que la primera line es un salto de linea
					line = f.readline()
					#print(len(line))
					if line == "" and not corrida:
						break
					line.strip('\n')
					data = line.split()
					if(len(data)==1):
						corrida = True
						#st.write(data[0] +"\n")
						renglones = ["MediaIni","MediaIS+-stdev", "MAccIni+-stdev",
									"MAccIS+-stdev", "Tiempo Medio"]
						string = '\t'.join(x for x in renglones)
						#st.write(string +"\n")
						continue
					elif len(data)==0:#RECORDAR ELIMINAR ITERACIONES DE LOS OTROS RESULTAOS
						results_data = np.array(matrix,dtype = 'float64')
						instances_results_data.append(results_data)
						print("Leido "+instance_name)
						break
						#print(results_data)
						#mediaIni = np.mean(results_data[:,0])
						#mediaIS = np.mean(results_data[:,1])/results_data[0,0]
						#mediaISStDev = np.std(results_data[:,1]/results_data[0,0],ddof = 1)
						#mAccIni = np.mean(results_data[:,2])
						#mAccIniStDev = np.std(results_data[:,2],ddof = 1) 
						#mAccIS = np.mean(results_data[:,3])
						#mAccISStDev = np.std(results_data[:,3], ddof = 1)
						#mTime = np.mean(results_data[:,4])
						#values = [mediaIni,mediaIS,mAccIni,mAccIS,mTime]
						#values = [x for x in map(str, values)]
						#values[1] = str(mediaIS)+"+-"+str(mediaISStDev)
						#values[2] = str(mAccIni)+"+-"+str(mAccIniStDev)
						#values[3] = str(mAccIS)+"+-"+str(mAccISStDev)
						#string = '\t'.join(x for x in values)
						#st.write(string+'\n')
						#matrix = []
						#corrida = False
						continue
					matrix.append(data)

				f.close()
				#st.close()
			if len(instances_results_data)==0:
				continue
			
			size_name = fd.replace("/","")
			meta_name = meta.replace("/","")
			reduction_data = list(map(lambda x:x[:,1]/x[0,0],instances_results_data))
			title = "Porcentaje de reducción para "+ \
					meta_name+\
					" en instancias "+size_name

			fig, ax1 = plt.subplots(figsize=(10, 6))
			fig.canvas.set_window_title(title)
			fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
			
			bp = ax1.boxplot(reduction_data, notch=0, sym='o', vert=1, whis=1.5)
			plt.setp(bp['boxes'], color='black')
			plt.setp(bp['whiskers'], color='black',linestyle='solid')
			plt.setp(bp['fliers'], color='red', marker='o')

			# Add a horizontal grid to the plot, but make it very light in color
			# so we can use it for reading data values but not be distracting
			ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
						   alpha=0.8)

			# Hide these grid behind plot objects
			ax1.set_axisbelow(True)
			ax1.set_title(title)
			ax1.set_xlabel('Instancia')
			ax1.set_ylabel('Porcentaje final')

			numBoxes = len(results)
			medians = list(range(numBoxes))
			for i in range(numBoxes):
				med = bp['medians'][i]
				# Finally, overplot the sample averages, with horizontal alignment
				# in the center of each box
				ax1.plot([np.average(med.get_xdata())], [np.average(reduction_data[i])],
						 color='w', marker='*', markeredgecolor='k')
			# Set the axes ranges and axes labels
			ax1.set_xlim(0.5, numBoxes + 0.5)
			top = 1.05
			bottom = int(min(list(map(lambda x : np.min(x),reduction_data)))*100)/100-0.05
			ax1.set_ylim(bottom, top)
			major_ticks = np.arange(bottom, top , 0.1 )
			minor_ticks = np.arange(bottom, top , 0.05)
			ax1.set_yticks(major_ticks)
			ax1.set_yticks(minor_ticks, minor=True)
			ax1.set_xticklabels(instances_names,
								rotation=0, fontsize=13)

			# Due to the Y-axis scale being different across samples, it can be
			# hard to compare differences in medians across the samples. Add upper
			# X-axis tick labels with the sample medians to aid in comparison
			# (just use two decimal places of precision)
			pos = np.arange(numBoxes) + 1

			meanTimes = list(map(lambda x: np.mean(x[:,4]),instances_results_data))
			upperLabels = [str(np.round(s,1))+"seg" for s in meanTimes]
			weights = ['bold', 'semibold']
			for tick, label in zip(range(numBoxes), ax1.get_xticklabels()):
				k = 1
				ax1.text(pos[tick], top - (top*0.05), upperLabels[tick],
						 horizontalalignment='center', size='x-small', weight=weights[k])
			fig.text(0.80, 0.015, '*', color='white', backgroundcolor='black',
					weight='roman', size='medium')
			fig.text(0.815, 0.015, ' Average Value', color='black', weight='roman',
					 size='x-small')

			plt.show()
			

			###########ACC#########
			
			size_name = fd.replace("/","")
			meta_name = meta.replace("/","")
			accuracy_data = list(map(lambda x:[x[:,2],x[:,3]],instances_results_data))
			#print(len(accuracy_data))
			accuracy_data = [acc for pair in accuracy_data for acc in pair]
			#print(len(accuracy_data))
			title = "Comparación de clasificación con "+ \
					meta_name+\
					" en instancias "+size_name

			fig, ax1 = plt.subplots(figsize=(10, 8))
			fig.canvas.set_window_title(title)
			fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
			
			bp = ax1.boxplot(accuracy_data, notch=0, sym='o', vert=1, whis=1.5)
			plt.setp(bp['boxes'], color='black')
			plt.setp(bp['whiskers'], color='black',linestyle='solid')
			plt.setp(bp['fliers'], color='red', marker='o')

			# Add a horizontal grid to the plot, but make it very light in color
			# so we can use it for reading data values but not be distracting
			ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
						   alpha=0.8)

			# Hide these grid behind plot objects
			ax1.set_axisbelow(True)
			ax1.set_title(title)
			ax1.set_xlabel('Instancia')
			ax1.set_ylabel('Porcentaje final')

			boxColors = ['white', 'royalblue']
			numBoxes = len(accuracy_data)
			medians = list(range(numBoxes))
			for i in range(numBoxes):
				box = bp['boxes'][i]
				boxX = []
				boxY = []
				for j in range(5):
					boxX.append(box.get_xdata()[j])
					boxY.append(box.get_ydata()[j])
				boxCoords = list(zip(boxX, boxY))
				# Alternate between Dark Khaki and Royal Blue
				k = i % 2
				boxPolygon = Polygon(boxCoords, facecolor=boxColors[k])
				ax1.add_patch(boxPolygon)
				# Now draw the median lines back over what we just filled in
				med = bp['medians'][i]
				medianX = []
				medianY = []
				for j in range(2):
					medianX.append(med.get_xdata()[j])
					medianY.append(med.get_ydata()[j])
					ax1.plot(medianX, medianY, 'k')
					medians[i] = medianY[0]
				# Finally, overplot the sample averages, with horizontal alignment
				# in the center of each box
				ax1.plot([np.average(med.get_xdata())], [np.average(accuracy_data[i])],
						 color='w', marker='*', markeredgecolor='k')
			# Set the axes ranges and axes labels
			ax1.set_xlim(0.5, numBoxes + 0.5)
			top = 1.05
			bottom = int(min(list(map(lambda x : np.min(x),accuracy_data)))*100)/100-0.05
			ax1.set_ylim(bottom, top)
			major_ticks = np.arange(bottom, top , 0.1 )
			minor_ticks = np.arange(bottom, top , 0.05)
			ax1.set_yticks(major_ticks)
			ax1.set_yticks(minor_ticks, minor=True)
			ax1.set_xticklabels(np.repeat(instances_names,2),
								rotation=45, fontsize=13)

			
			fig.text(0.80, 0.08, 'Precisión Inicial',
					backgroundcolor=boxColors[0], color='black', weight='roman',
					size='x-small')
			fig.text(0.80, 0.045, 'Precisión Final',
					 backgroundcolor=boxColors[1],
					 color='white', weight='roman', size='x-small')
			fig.text(0.80, 0.015, '*', color='white', backgroundcolor='black',
					weight='roman', size='medium')
			fig.text(0.815, 0.015, ' Average Value', color='black', weight='roman',
					size='x-small')



			plt.show()
			