
import os, sys
from random import randint

import numpy as np
from six.moves import cPickle as pickle

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

import matplotlib.pyplot as plt

CATEGORIAS = ["ARBOL", "GENTE", "VEHICULO"]
use_image = sys.argv[1] if len(sys.argv)>1 else False

################################
# Cargamos la imagen a predecir
################################

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

if use_image:
    print("Prediciendo imagen {0}".format(use_image))
    filepath = "imagenes/{0}".format(use_image)
    import cv2
    img = cv2.imread(filepath)
    img = cv2.resize(img, (32,32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
else:
    test = unpickle('data/test')
    data_x_test = test[b'data']

    # Seleccionamos la parte que nos interesa
    meta = unpickle('data/meta')
    labels = meta[b'fine_label_names']
    subcategorias = [b'oak_tree', b'pine_tree', b'palm_tree',b'girl', b'boy', b'woman', b'man', b'streetcar', b'motorcycle', b'bus']
    subcategorias = [labels.index(item) for item in subcategorias]
    indexes_test = np.where(np.isin(test[b'fine_labels'], subcategorias))
    data_x_test = data_x_test[indexes_test]

    # Y finalmente la imagen
    img = data_x_test[randint(0,len(data_x_test))].reshape(3,32,32).transpose(1,2,0)


#######################################
# Recuperamos el modelo
#######################################

model = load_model("generated/modelo")
prediction = model.predict(np.expand_dims(img, axis=0))[0] # # (batch of size 1)
prediction = np.round(prediction * 100, 2)

# Asociamos cada resultado a su categoria
predicciones_diccionario = {}
for i, categoria in enumerate(CATEGORIAS):
    predicciones_diccionario[categoria] = prediction[i]

plt.imshow(img)
plt.title(predicciones_diccionario)
plt.show()
