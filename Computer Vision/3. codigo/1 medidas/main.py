

import numpy as np
from sklearn import linear_model

###########################
# Leemos los datos
###########################

f = open("Untitled form.csv")
feature_labels = f.readline().rstrip('\n').split(',')

# cargamos las columnas que queremos
dataset_label = '¿A qué datos perteneces?'
feature_labels_x = ['Edad','Altura (cm)','Peso (kg)','Cintura (cm)']
feature_label_y = 'Sexo biológico'

dataset = np.loadtxt(
            f,
            dtype='str',
            delimiter=',',
            usecols=[feature_labels.index(c) for c in [dataset_label] + feature_labels_x + [feature_label_y]]
            )

# mezclamos los datos
np.random.shuffle(dataset)

# separamos el dataset en x e y
data_type = dataset[:,:1].reshape(-1)
data_x = dataset[:, 1:-1]
data_y = dataset[:, -1:].reshape(-1)
data_y = data_y == 'Femenino' # Codificamos la salida (Femenino=1, Masculino=0)

# separamos train y test sets
train_set_indexes = np.where(data_type=='Entrenamiento')
test_set_indexes = np.where(data_type=='Test')

data_x_train = data_x[train_set_indexes].astype('float')
data_y_train = data_y[train_set_indexes].astype('float')

data_x_test = data_x[test_set_indexes].astype('float')
data_y_test = data_y[test_set_indexes].astype('float')


######################################################
# Estamos listos para definir y entrenar el modelo!
######################################################

logreg = linear_model.LogisticRegression()
logreg.fit(data_x_train, data_y_train)

# Ahora comprobamos la eficacia, y nada más!!!
predicciones = logreg.predict(data_x_test)
print("Eficacia (reg. lineal) del {0}".format(np.mean(predicciones==data_y_test) * 100))



######################################################
# Red neuronal
######################################################


from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

modelo = MLPClassifier(random_state=1, verbose=False)

param_grid = [
  {
  	'hidden_layer_sizes': [(2,),(4,),(8,),(16,)], 
  	'solver':['lbfgs'],
  	# 'alpha': 10.0 ** -np.arange(1, 7),
  	# 'max_iter': [500,1000,1500]
  },
  {
  	'hidden_layer_sizes': [(2,),(4,),(8,),(16,)], 
  	'solver':['adam'],
  	# 'alpha': 10.0 ** -np.arange(1, 7),
  	# 'max_iter': [500,1000,1500]
  }
]

modelo_optimizado = GridSearchCV(modelo, param_grid, n_jobs=2, verbose=0)
modelo_optimizado.fit(data_x_train, data_y_train)

# print(pipe.grid_scores_)

data_y_predicho = modelo_optimizado.predict(data_x_test)


print("Eficacia (MLP): {}".format(100*np.mean(data_y_predicho==data_y_test)))