#!usr/bin/env python

import numpy as np
import csv
import re
from sklearn.preprocessing import LabelBinarizer
# from sklearn.pipeline import Pipeline


def limpia_texto(texto):
	return ' '.join(
		[palabra.lower() for palabra in 
			re.sub(
				"(@[A-Za-z0-9]+)|([A-Za-zá-ú]*[0-9]+[A-Za-zá-ú]*)|([^A-Za-z \tá-ú])|(\w+:\/\/\S+)", 
				" ", 
				texto
			).split()]
	)


def one_hot(input_, bolsa, invierte=False):
	if not invierte:
		onehot = np.zeros(len(bolsa))
		if input_ in bolsa:
			indice = bolsa.index(input_)
			onehot[indice] = 1
		return onehot
	else:
		indice = np.flatnonzero(input_==1)
		return None if len(indice)!=1 else bolsa[indice[0]]



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



# for row in data:
# 	print(row)


bolsa_de_palabras = sorted(set(' '.join(np.asarray(data)[:,4]).split()))



# onehot = one_hot("viernes", bolsa_de_palabras)


# print(one_hot(onehot, bolsa_de_palabras, invierte=True))


for document in data:

	texto = document[4]
	print(texto)
	
	# representacion = np.sum([one_hot(palabra) for palabra in document[4]])