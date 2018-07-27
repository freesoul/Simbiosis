#!usr/bin/env python

import re



#####################################################################
#
#	Funcion que hace el apaño de sacar la raiz de las palabras
#
#####################################################################


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