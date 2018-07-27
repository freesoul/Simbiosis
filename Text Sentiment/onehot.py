
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

