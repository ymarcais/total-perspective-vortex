import numpy as np

class Ica:

	def __init__(self) -> None:
		pass

	#center data X to axis=0
	def center_data(self, X):
		return X - np.mean(X, axis=0)
	
	# Whitening: transforming in such a way that the features become uncorrelated and have unit variance. 
	# Perform PCA and whiten the data , cov(X,Y)=(1/n-1) * ​∑​(xi​−Xˉ)(yi​−Yˉ)
	# SVD decomposes one matrix into 3 matrix (1- unitary, 2-diagonal, 3-transpose of unitary matrix)
	def whiten_data(self, X):
		cov_matrix = np.cov(X.T)
		_, S, V = np.linalg.svd(cov_matrix)
		whitened_X = X.dot(V / np.sqrt(S))
		return whitened_X
	
	#Activation tanh hyperbolic tangeant
	def tanh_(self, z):
		tanh = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
		return tanh
	

	# Tanh derivative
	def tanh_prime(self, z):
		return 1 -self.tanh_(z)**2
	

	#my ICA
	def fast_ica(self, X, n_components, max_iterations, tolerance=1e-6):
		X = self.center_data(X)
		X = self.whiten_data(X)
		_, n_features = X.shape

		# Initialize random weights
		np.random.seed(42)
		W = np.random.rand(n_components, n_features)

		for iterations in range(max_iterations):
			S = W @ X
			G = self.tanh_(S)

			W_new = (X @ G.T) / len(X.T) - np.mean(self.tanh_prime(S), axis=1)[:, np.newaxis] * W.sum(axis=1)
			W = W_new
		
		return W @ X
	

def main():
	ica = Ica()
	max_iterations = 1000 

if __name__ == "__main__":
	main()

						
				
				


	



	

	



