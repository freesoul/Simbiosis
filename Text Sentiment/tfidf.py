#!usr/bin/env python


import re
import numpy as np
import csv

np.set_printoptions(threshold=np.nan)



#####################################################################
#
#	
#
#####################################################################

def limpia_texto(texto):
	return ' '.join(
		[palabra.lower() for palabra in 
			re.sub(
				"(@[A-Za-z0-9]+)|([A-Za-zá-ú]*[0-9]+[A-Za-zá-ú]*)|([^A-Za-z \tá-ú])|(\w+:\/\/\S+)", 
				" ", 
				texto
			).split()]
	)




#####################################################################
#
#	
#
#####################################################################

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





#####################################################################
#
#	Leemos el dataset
#
#####################################################################


with open('dataset.csv', 'r') as f:
	raw = csv.reader(f, delimiter=',', quotechar='"')

	data = []
	for row in raw:
		data.append([
			row[0],
			row[1],
			limpia_texto(row[0])
		])

tweets_limpios = np.asarray(data)[:,2]
Y = np.asarray(data)[:,1]




#####################################################################
#
#	
#
#####################################################################


bolsa_de_palabras = sorted(set(' '.join(tweets_limpios).split()))

# print(bolsa_de_palabras)
# print("Hay {} palabras".format(len(bolsa_de_palabras)))



#####################################################################
#
#	IDF a mano
#
#####################################################################

tweets_limpios = tweets_limpios[:10] # para que no pete


def en_cuantos_documentos(palabra, data_):
	return len([1 for row in data_ if palabra in row.split()])

idfs = np.array([en_cuantos_documentos(palabra, tweets_limpios) for palabra in bolsa_de_palabras])
idfs = np.log(len(tweets_limpios)) - np.log(1 + idfs)
tfs = np.array([np.sum([one_hot(palabra, bolsa_de_palabras) for palabra in tweet.split()], axis=0) / len(tweet.split()) for tweet in tweets_limpios])
tfidfs = tfs*idfs

with open("tfidfs.npy", "wb") as f:
	np.save(f, tfidfs)

# with open("tfidfs.npy", "rb") as f:
# 	tfidfs = np.load(f)

print(tfidfs[1])






#####################################################################
#
#	IDF con sklearn
#
#####################################################################

# from sklearn.feature_extraction.text import TfidfVectorizer


# vectorizer = TfidfVectorizer()
# tfidfs = vectorizer.fit_transform(tweets_limpios)
