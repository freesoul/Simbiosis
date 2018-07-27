
import os

import numpy as np
from six.moves import cPickle as pickle

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D


##############################
# Cargamos los archivos
##############################

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

meta = unpickle('data/meta')
train = unpickle('data/train')
test = unpickle('data/test')

labels = meta[b'fine_label_names']


#######################################
# Extraemos las categorias que queremos
#######################################

# Extraemos los indices de los de los metadatos
trees = [labels.index(item) for item in [b'oak_tree', b'pine_tree', b'palm_tree']] # 0
people = [labels.index(item) for item in [b'girl', b'boy', b'woman', b'man']] # 1
vehicles = [labels.index(item) for item in [b'streetcar', b'motorcycle', b'bus']] # 2

# Extraemos las filas de datos que queremos del train y test set
indexes_train = np.where(np.isin(train[b'fine_labels'], trees+people+vehicles))
indexes_test = np.where(np.isin(test[b'fine_labels'], trees+people+vehicles))

data_x_train = train[b'data'][indexes_train]
data_y_train = np.array(train[b'fine_labels'])[indexes_train]
data_x_test = test[b'data'][indexes_test]
data_y_test = np.array(test[b'fine_labels'])[indexes_test]

# Homogeineizamos las clases con un nuevo codigo
np.place(data_y_train, np.isin(data_y_train, trees), 0)
np.place(data_y_train, np.isin(data_y_train, people), 1)
np.place(data_y_train, np.isin(data_y_train, vehicles), 2)

np.place(data_y_test, np.isin(data_y_test, trees), 0)
np.place(data_y_test, np.isin(data_y_test, people), 1)
np.place(data_y_test, np.isin(data_y_test, vehicles), 2)


# Contamos los elementos (solo para seguimiento)
unique, counts = np.unique(data_y_train, return_counts=True)
print("Tamaño de las categorías (train): ", dict(zip(unique, counts)))

unique, counts = np.unique(data_y_test, return_counts=True)
print("Tamaño de las categorías (test): ", dict(zip(unique, counts)))


################################
# Mezclamos los datos
################################

perm = np.random.permutation(data_x_train.shape[0])
data_x_train = data_x_train[perm]
data_y_train = data_y_train[perm]

perm = np.random.permutation(data_x_test.shape[0])
data_x_test = data_x_test[perm]
data_y_test = data_y_test[perm]



#######################################
# Reconfiguramos los datos
#######################################
data_x_train = data_x_train.reshape(-1,3,32,32).transpose(0,2,3,1)
data_x_test = data_x_test.reshape(-1,3,32,32).transpose(0,2,3,1)


#######################################
# Codificamos la y
#######################################
data_y_train = keras.utils.to_categorical(data_y_train, 3)
data_y_test = keras.utils.to_categorical(data_y_test, 3)



#######################################
# Definimos el modelo
#######################################


model = Sequential()

# CONVOLUCION. Tamaño final: 29x29x32
model.add(Conv2D(   32,
                    (4, 4),
                    padding='same',
                    input_shape=data_x_train.shape[1:]
        ))

model.add(Activation('relu'))


# CONVOLUCION. Tamaño final: 26x26x32
model.add(Conv2D(   32,
                    (4, 4),
                    padding='same',
                    input_shape=data_x_train.shape[1:]
        ))

model.add(Activation('relu'))


# MAX POOLING. Tamaño final: 13x13x32
model.add(MaxPooling2D(pool_size=(2,2)))

# Dropout para prevenir overfitting
model.add(Dropout(0.25))


# CONVOLUCION. Tamaño final: 10x10x64
model.add(Conv2D(   64,
                    (4, 4),
                    padding='same',
                    input_shape=data_x_train.shape[1:]
        ))

model.add(Activation('relu'))

# MAX POOLING. Tamaño final: 5x5x64
model.add(MaxPooling2D(pool_size=(2,2)))

# Dropout para prevenir overfitting
model.add(Dropout(0.25))

# Capa totalmente conectada. Número de neuronas: 1600
model.add(Flatten())

# Capa de salida: 3 neuronas
model.add(Dense(3))
model.add(Activation('softmax'))


#######################################
# Compilamos el modelo y entrenamos
#######################################


# Definimos el optimizador
optimizador = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

#Compilamos y entrenamos el modelo
model.compile(loss='categorical_crossentropy',
              optimizer=optimizador,
              metrics=['accuracy'])

model.fit(data_x_train, data_y_train, verbose=True, epochs=20, batch_size=100)



#######################################
# Guardamos el modelo entrenado
# (Por si lo queremos reutilizar)
#######################################


if not os.path.isdir("generated"):
    os.makedirs("generated")
model_path = os.path.join("generated", "modelo")
model.save(model_path)
print('Saved trained model at %s ' % model_path)



########################################
# Comprobamos la eficacia en el test set (~80% en 20 epochs con batches de 100)
########################################

scores = model.evaluate(data_x_test, data_y_test, verbose=1)
print('Loss:', scores[0])
print('Eficacia:', scores[1])
