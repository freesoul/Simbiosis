#!usr/bin/env python

import numpy as np
import csv
import re
from sklearn.preprocessing import LabelBinarizer
import pickle


from limpia_texto import *
from lemmatizador import *



#####################################################################
#
#	
#
#####################################################################

columnas = {
	'user':0,
	'fecha':3,
	'likes':5,
	'texto':8
}

with open('twits.csv', 'r') as f:
	raw = list(csv.reader(f, delimiter=',', quotechar='"'))[1:]

	data = []
	for row in raw:
		data.append([
			row[columnas['user']],
			row[columnas['fecha']].split(" ")[0],
			int(row[columnas['likes']]),
			row[columnas['texto']],
			limpia_texto(row[columnas['texto']])
		])

data = np.array(data)


#####################################################################
#
#	
#
#####################################################################



bolsa_de_palabras = sorted(set(' '.join(np.asarray(data)[:,4]).split()))




#####################################################################
#
#	
#
#####################################################################

with open("modelos/nb.model", "rb") as f:
	naivebayes = pickle.load(f)

predicciones = naivebayes.predict(data[:,4])


#####################################################################
#
#	Concatenamos
#
#####################################################################

data = np.concatenate([data, predicciones.reshape(-1,1)], axis=1)


#####################################################################
#
#	Sacamos datos
#
#####################################################################

total = len(data)
print("Total: {}".format(total))

positivos = np.sum(data[:,5]=='P')
print("Positivos: {} ({})".format(positivos, positivos/total))

negativos = total - positivos
print("Negativos: {} ({})".format(negativos, negativos/total))



#####################################################################
#
#	Grafica
#
#####################################################################

print("Primera mención: {}".format(data[:,1][-1]))
print("Última mención: {}".format(data[:,1][0]))


# Vamos a representar todos los meses de 2010 a 2018

fechas = ['{}-{num:02d}'.format(anio,num=mes) for anio in range(2010,2019) for mes in range(1,13)]
positivos = np.zeros(len(fechas))
negativos = np.zeros(len(fechas))

# Contamos

for i,sentimiento in enumerate(data[:,5]):
	current_key = data[:,1][i][:7]
	index = fechas.index(current_key)
	if sentimiento=='P': 
		positivos[index]+=1
	elif sentimiento=='N':
		negativos[index]+=1

proporciones = np.divide(positivos, positivos+negativos, out=np.full(len(fechas), None), where=(positivos+negativos)!=0)
totales = positivos+negativos


import matplotlib.cbook as cbook
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


dates = np.array(fechas).astype('O')
fig, ax = plt.subplots()
ax.set_title('Timeline de sentimientos de tweets con LaCasaInvisible')

ax.bar(np.arange(len(fechas)), totales, align='center', alpha=1, color="C3", width=1)
ax.bar(np.arange(len(fechas)), positivos, align='center', alpha=1, color="C2", width=1)

ax.set_xticks(np.arange(len(fechas))) # fecha for i, fecha in enumerate(fechas) if i%4==0
ax.set_xticklabels([fecha if totales[i]>0 else '' for i, fecha in enumerate(fechas)])


#####################################################################
#
#	Regresion
#
#####################################################################
indexes = np.where(positivos!=None)
positivos = positivos[indexes]


from sklearn.linear_model import LinearRegression

reg  = LinearRegression()
reg.fit(indexes[0].reshape(-1,1), positivos.reshape(-1,1))

reg_x = np.arange(len(fechas))
reg_y = reg.predict(np.arange(len(fechas)).reshape(-1,1)).ravel()


ax.plot(reg_x, reg_y)

# ax.bar(np.arange(len(fechas))+0.5, positivos, align='center', alpha=1, color="C2", width=1)
# ax.bar(np.arange(len(fechas))-0.5, negativos, align='center', alpha=1, color="C3", width=1)

# ax.set_xticks(np.arange(len(fechas))) # fecha for i, fecha in enumerate(fechas) if i%4==0
# ax.set_xticklabels([fecha if totales[i]>0 else '' for i, fecha in enumerate(fechas)])

fig.autofmt_xdate()
plt.show()




