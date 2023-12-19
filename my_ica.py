import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
import mne
from sklearn.decomposition import FastICA
from mne.preprocessing import ICA
from mne.datasets import sample

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
	

	def fast_ica_sklearn(self, epochs, n_components, max_iterations=500, tol=1e-4):
		X = epochs.get_data()
		X = self.center_data(X)
		X = self.whiten_data(X)

		n_epochs, n_channels, n_samples = X.shape
		#X_reshaped = X.reshape( n_samples, n_components * n_channels)
		X_reshaped = X.flatten()
		X_reshaped = X.reshape(-1, 1)
		print("X X_reshape shape", X_reshaped.shape)


		data = FastICA(n_components=n_components, max_iter=max_iterations, tol=tol, random_state=42, whiten='unit-variance')
		modified_data = data.fit_transform(X_reshaped)
		learned_components = data.components_
		mixing_matrix = data.mixing_
		print("learn_components shape", learned_components.shape)
		print("mixing matrix shape", mixing_matrix.shape)
		modified_data = modified_data.reshape(n_epochs, n_samples, n_channels)

		print("modified data shape", modified_data.shape)
		print("epochs len", len(epochs))
		modified_info = mne.create_info(n_components, epochs.info['sfreq'], ch_types='misc')


		print("modified data shape", modified_data.shape)
		print("modified info shape", len(modified_info))
		print("channel data shape", modified_data.T[:, :, np.newaxis].shape)
    
		print("modified info shape", len(modified_info))
		print("channel data shape", modified_data.T[:, :, np.newaxis].shape)
		#A = modified_data.T[:, :, np.newaxis]
		#modified_data = A.transpose(0, 1, 2)

		print("modified_data", modified_data.shape)
		print("modified_info", len(modified_info))
		print("n_components", n_components)
		fastica_epochs = mne.EpochsArray(modified_data, modified_info, epochs.events)

		return fastica_epochs
	
	#Activation tanh hyperbolic tangeant
	def tanh_(self, z):
		return np.tanh(0.5 * z)
	

	# Tanh derivative
	def tanh_prime(self, z):
		cosh_z = np.cosh(0.5 * z)
		return 0.5 / cosh_z**2
	
	#my ICA
	def fast_ica(self, epochs, n_components, learning_rate=0.004, max_iterations=500, tol=1e-5):

		X = epochs.get_data()
		X = self.center_data(X)
		X = self.whiten_data(X)

		n_epochs, n_samples, n_channels = X.shape

		X_reshaped = X.transpose(1, 0, 2).reshape(n_samples, -1)
		scaler = StandardScaler()
		X_standardized_reshaped = scaler.fit_transform(X_reshaped)
		X = X_standardized_reshaped.reshape(n_samples, n_channels, n_epochs).transpose(1, 0, 2)

		X = zscore(X)
		ica_epochs_data = np.empty((n_epochs, n_components, n_channels))

		np.random.seed(42)
		W = np.eye(n_components, n_channels)
		print("W", W.shape)

		for iterations in range(max_iterations):
			
			for epoch_idx in range(n_epochs):
				S = W @ X[:, :, epoch_idx]
				G = self.tanh_(S)

				A = np.diag(self.tanh_prime(S))
				A = A.reshape(1, -1)
				delta_W = (1 / n_samples) * (X[:, :, epoch_idx] @ G.T - (A @ W).T)

				W = W + learning_rate * delta_W.T

				ica_epochs_data[epoch_idx, :, :] = W @ (X[:, :, epoch_idx]).T

			if np.linalg.norm(delta_W) < tol:
				print(f"Converged after {iterations + 1} iterations.")
				break
		
		modified_data = ica_epochs_data.transpose(0, 1, 2)
		# Normalize independent components to unit variance
		for i in range(n_components):
			modified_data[:, i, :] /= np.std(modified_data[:, i, :])

		modified_info = mne.create_info(n_components, epochs.info['sfreq'], ch_types='eeg')
		modified_epochs = mne.EpochsArray(modified_data, modified_info, epochs.events)
		modified_data = ica_epochs_data.transpose(1, 0, 2)		
		
		events = modified_epochs.events

		return modified_epochs, events
	

def main():

	max_iterations = 500
	ica = My_ica()

if __name__ == "__main__":
	main()
