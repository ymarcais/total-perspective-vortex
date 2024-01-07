import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
import mne
from sklearn.decomposition import FastICA
from mne.preprocessing import ICA
from mne.datasets import sample
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler

class My_ica:

	def __init__(self) -> None:
		pass

	#center data X to axis=0
	def center_data(self, X):
		return X - np.mean(X, axis=0)
	
	# Whitening: transforming in such a way that the features become uncorrelated and have unit variance. 
	# Perform PCA and whiten the data , cov(X,Y)=(1/n-1) * ​∑​(xi​−Xˉ)(yi​−Yˉ)
	# SVD decomposes one matrix into 3 matrix (1- unitary, 2-diagonal, 3-transpose of unitary matrix)
	def whiten_data(self, X):
		#X = zscore(X, axis=1)
		n_channels, n_samples, n_epochs = X.shape
		X_reshaped = X.transpose(1, 0, 2).reshape(n_samples, -1)
		cov_matrix = np.cov(X_reshaped.T)
		_, S, V = np.linalg.svd(cov_matrix)
		whitened_X_reshaped = X_reshaped.dot(V / np.sqrt(S))
		whitened_X = whitened_X_reshaped.reshape(n_samples, n_channels, n_epochs).transpose(1, 0, 2)
    
		return whitened_X
	

	def fast_ica_mne(self, epochs, n_components, max_iterations=1000, tol=1e-5):
		
		print(mne.__file__)
		ica = ICA(n_components=n_components, max_iter=max_iterations, method='picard')
		ica.fit(epochs)
		get_data = epochs.get_data()
		get_data = zscore(get_data, axis=0)
		cleaned_data = ica.apply(mne.EpochsArray(get_data, epochs.info))
		cleaned_data = cleaned_data.get_data()
		print("cleaned data", cleaned_data.shape)
		cleaned_data = zscore(cleaned_data, axis=0)
		extracted_components = ica.get_components()
		print("mne components shape", extracted_components.shape)
		
		return 	extracted_components
	
	#Activation tanh hyperbolic tangeant
	def tanh_(self, z):
		return np.tanh(0.5 * z)


	# Tanh derivative
	def tanh_prime(self, z):
		cosh_z = np.cosh(0.5 * z)
		return 0.5 / cosh_z**2
	
	#my ICA
	def fast_ica(self, epochs, n_components, learning_rate=0.02, max_iterations=300, tol=1e-4):

		X = epochs.get_data()
		X = self.center_data(X)
		X = self.whiten_data(X)

		n_epochs, n_channels, n_samples = X.shape

		X_reshaped = X.transpose(1, 0, 2).reshape(n_channels, -1)
		#scaler = StandardScaler()
		#X = scaler.fit_transform(X_reshaped)
		X = zscore(X_reshaped, axis=0)

		np.random.seed(42)
		W = np.eye(n_components, n_channels)
	
		for iterations in range(max_iterations):
				
			S = W @ X
			scaler = MinMaxScaler()
			S = scaler.fit_transform(S)
			G = self.tanh_(S)
			A = np.diag(self.tanh_prime(S))
			A = A.reshape(1, -1)
			delta_W = (1 / n_samples) * (X @ G.T - (A @ W).T)

			W += learning_rate * delta_W.T

			if np.linalg.norm(delta_W) < tol:
				print(f"Converged after {iterations + 1} iterations.")
				break
		print(" 0000 W shape", W.shape)
		
		return W
	

def main():

	max_iterations = 500
	ica = My_ica()

if __name__ == "__main__":
	main()
