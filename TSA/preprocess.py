#!usr/bin/env python

import numpy as np
import csv
import re



def limpia_texto(texto):
	return ' '.join(
		[palabra.lower() for palabra in 
			re.sub(
				"(@[A-Za-z0-9]+)|([A-Za-zá-ú]*[0-9]+[A-Za-zá-ú]*)|([^A-Za-z \tá-ú])|(\w+:\/\/\S+)", 
				" ", 
				texto
			).split()]
	)




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

print(bolsa_de_palabras)