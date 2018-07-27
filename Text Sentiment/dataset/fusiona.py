#!usr/bin/env python


import re
import xml.etree.ElementTree
import csv
import numpy as np
import random


#####################################################################
#
#	Datos xml
#
#####################################################################

data_xml_raw = xml.etree.ElementTree.parse('4c.xml').getroot()

data_xml = []
for entry in data_xml_raw:
	data_xml.append([
		entry.find('content').text,
		entry.find('sentiment').find('polarity').find('value').text
	])

print("XML len: {}".format(len(data_xml)))


#####################################################################
#
#	Datos csv
#
#####################################################################

with open('2c.csv', 'r') as f:
	data_csv_raw = list(csv.reader(f, delimiter=',', quotechar='"'))[1:]

	data_csv = []
	for row in data_csv_raw:
		data_csv.append([
			row[0],
			'P' if row[1]=='pos' else 'N'
		])

print("CSV len: {}".format(len(data_csv)))


#####################################################################
#
#	Datos totales
#
#####################################################################

data = data_xml + data_csv

print("Full len: {}".format(len(data)))



#####################################################################
#
#	Eliminamos NEU y NONE
#
#####################################################################

data = [row for row in data if row[1] not in ['NONE', 'NEU']]

print("Full len (sin NEU ni NONE): {}".format(len(data)))



#####################################################################
#
#	Mezclamos y guardamos
#
#####################################################################

random.shuffle(data)

with open("dataset.csv", 'w') as f:
	handler = csv.writer(f, quoting=csv.QUOTE_ALL)
	for row in data:
		handler.writerow(row)

