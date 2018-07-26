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



data_xml = xml.etree.ElementTree.parse('intertass/intertass-ES-train-tagged.xml').getroot()

data = []
for entry in data_xml:
	data.append([
		entry.find('content').text,
		limpia_texto(entry.find('content').text),
		entry.find('sentiment').find('polarity').find('value').text
	])


tweets_limpios = np.asarray(data)[:,1]
Y = np.asarray(data)[:,2]



data_xml_test = xml.etree.ElementTree.parse('intertass/intertass-ES-development-tagged.xml').getroot()

data_test = []
for entry in data_xml_test:
	data_test.append([
		entry.find('content').text,
		limpia_texto(entry.find('content').text),
		entry.find('sentiment').find('polarity').find('value').text
	])


tweets_limpios_test = np.asarray(data_test)[:,1]
Y_test = np.asarray(data_test)[:,2]




from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


with open('stopwords.txt','r') as f:
	stopwords = [line.rstrip() for line in f.readlines()]





def Lemmatizador(texto, minWordLen=4):

	tokens = texto.split()

	output = []
	for palabra in tokens:
		corregida = palabra

		regla1 = r'((i[eé]ndo|[aá]ndo|[aáeéií]r|[^u]yendo)(sel[ao]s?|l[aeo]s?|nos|se|me))'
		step1 = re.search(regla1, corregida)
		if step1:
		    if (len(palabra)-len(step1.group(1))) >= minWordLen:
		        corregida = corregida[:-len(step1.group(1))]
		    elif (len(palabra)-len(step1.group(3))) >= minWordLen:
		        corregida = corregida[:-len(step1.group(3))]

		regla2 = {
		  '(anzas?|ic[oa]s?|ismos?|[ai]bles?|istas?|os[oa]s?|[ai]mientos?)$' : '',
		  '((ic)?(adora?|ación|ador[ae]s|aciones|antes?|ancias?))$' : '',
		  '(log[íi]as?)$' : 'log',
		  '(ución|uciones)$' : 'u',
		  '(encias?)$' : 'ente',
		  '((os|ic|ad|(at)?iv)amente)$' : '',
		  '(amente)$' : '',
		  '((ante|[ai]ble)?mente)$' : '',
		  '((abil|ic|iv)?idad(es)?)$' : '',
		  '((at)?iv[ao]s?)$' : '',
		  '(ad[ao])$' : '',
		  '(ando)$' : '',
		  '(aci[óo]n)$' : '',
		  '(es)$' : ''
		}
		for key in regla2:
		    tmp = re.sub(key, regla2[key], corregida)
		    if tmp!=corregida and len(tmp)>=minWordLen:
		        corregida = tmp

		regla3 = {
		'(y[ae]n?|yeron|yendo|y[oó]|y[ae]s|yais|yamos)$',
		'(en|es|éis|emos)$',
		'(([aei]ría|ié(ra|se))mos)$',
		'(([aei]re|á[br]a|áse)mos)$',
		'([aei]ría[ns]|[aei]réis|ie((ra|se)[ns]|ron|ndo)|a[br]ais|aseis|íamos)$',
		'([aei](rá[ns]|ría)|a[bdr]as|id[ao]s|íais|([ai]m|ad)os|ie(se|ra)|[ai]ste|aban|ar[ao]n|ase[ns]|ando)$',
		'([aei]r[áé]|a[bdr]a|[ai]d[ao]|ía[ns]|áis|ase)$',
		'(í[as]|[aei]d|a[ns]|ió|[aei]r)$',
		'(os|a|o|á|í|ó)$',
		'(u?é|u?e)$',
		'(ual)$',
		'([áa]tic[oa]?)$'
		}
		for pattern in regla3:
		    tmp = re.sub(pattern, '', corregida)
		    if tmp!=corregida and len(tmp)>=minWordLen:
		        corregida = tmp

		output.append(corregida)
	return ' '.join(output)




#  TfidfVectorizer(),
pipe = make_pipeline(CountVectorizer(ngram_range=(1,3), preprocessor=Lemmatizador, stop_words=stopwords),  MultinomialNB())

pipe.fit(tweets_limpios, Y)

Y_pred = pipe.predict(tweets_limpios_test)


print(np.mean(Y_pred==Y_test))
