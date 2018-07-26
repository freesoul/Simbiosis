#!usr/bin/env python


import re
import xml.etree.ElementTree
import numpy as np

np.set_printoptions(threshold=np.nan)



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




data_xml = xml.etree.ElementTree.parse('intertass/intertass-ES-train-tagged.xml').getroot()

data = []
for entry in data_xml:
	data.append([
		entry.find('content').text,
		limpia_texto(entry.find('content').text),
		entry.find('sentiment').find('polarity').find('value').text
	])


tweets_limpios = np.asarray(data)[:,1]

bolsa_de_palabras = sorted(set(' '.join(tweets_limpios).split()))

# print(bolsa_de_palabras)

def en_cuantos_documentos(palabra, data_):
	return len([1 for row in data_ if palabra in row.split()])

idfs = np.array([en_cuantos_documentos(palabra, tweets_limpios) for palabra in bolsa_de_palabras])
# # print(idfs)

idfs = np.log(len(tweets_limpios)) - np.log(1 + idfs)
# # idfs = np.log(idfs)
# # idfs = idfs - np.log(len(tweets_limpios))

# tfs = np.array([np.sum([one_hot(palabra, bolsa_de_palabras) for palabra in tweet.split()], axis=0) / len(tweet.split()) for tweet in tweets_limpios])

# tfidfs = tfs*idfs

# with open("tfidfs.npy", "wb") as f:
# 	np.save(f, tfidfs)

with open("tfidfs.npy", "rb") as f:
	tfidfs = np.load(f)

# print(tfidfs[0])

Y = np.asarray(data)[:,2]



from sklearn.naive_bayes import MultinomialNB

modelo = MultinomialNB()
modelo.fit(tfidfs, Y)



#########################
# test set
########################

data_xml_test = xml.etree.ElementTree.parse('intertass/intertass-ES-development-tagged.xml').getroot()

data_test = []
for entry in data_xml_test:
	data_test.append([
		entry.find('content').text,
		limpia_texto(entry.find('content').text),
		entry.find('sentiment').find('polarity').find('value').text
	])


tweets_limpios_test = np.asarray(data_test)[:,1]



tfs_test = np.array([np.sum([one_hot(palabra, bolsa_de_palabras) for palabra in tweet.split()], axis=0) / len(tweet.split()) for tweet in tweets_limpios_test])
X_test = tfs_test*idfs



Y_test = np.asarray(data_test)[:,2]

Y_pred = modelo.predict(X_test)



print(np.mean(Y_test==Y_pred)*100)

# print(en_cuantos_documentos("viernes", tweets_limpios))

# idfs = np.array([1+])


# print(np.array(data)[:,1])

# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer

# cv = CountVectorizer()
# data2 = cv.fit_transform(np.array(data)[:,1])

# print(data2.shape)
# print(data2[0])
# print(np.array(data)[:,1][0])
# print(cv.vocabulary_.get(u'me'))


# # bolsa_de_palabras = sorted(set(' '.join(np.asarray(data)[:,1]).split()))


# # # np.set_printoptions(threshold=np.nan)
# # for document in data:

# # 	tweet = document[1]
# # 	# print(tweet)
# # 	representacion = np.sum([one_hot(palabra, bolsa_de_palabras) for palabra in tweet.split()], axis=0)
# # 	# print(representacion)

# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(data2)
# print(X_train_tfidf.shape)
# print(X_train_tfidf[2])