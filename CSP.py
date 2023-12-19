import numpy as np
from scipy import linalg
from sklearn.base import TransformerMixin, BaseEstimator



class CSP(TransformerMixin, BaseEstimator):
	def __init__(self, n_components=5):
		if not isinstance(n_components, int):
			raise ValueError('n_components must be an integer.')
		
		self.n_components =  n_components
	
	def _compute_covariance_matrices(self, X, y):
		print("y compute shape", y.shape)
		y = np.squeeze(y)
		print("y squeez shape", y.shape)
		print("xxxX shape0", X.shape)
		print("xxxn_components", self.n_components)
		X_reshaped = X.reshape(X.shape[0], 64, 64)
		print("xxxX shape1", X_reshaped.shape)
		_, n_channels, _ =X_reshaped.shape

		covs = []
		for cur_class in self._classes:
			"""Concatenate epochs before computing the covariance."""
			print("y==cur_class shape:", (y==cur_class).shape)
			print("X_reshaped shape", X_reshaped.shape)

			x_class =  X_reshaped[y==cur_class]
			print("x_class shape:", x_class.shape)

			#x_class = np.transpose(x_class, [1, 0, 2])
			x_class = x_class.reshape(n_channels, -1)
			cov_mat = np.cov(x_class)
			covs.append(cov_mat)
		
		return np.stack(covs)

	def fit(self, X, y):
		"""Estimate the CSP decomposition on epochs."""


		self._classes = np.unique(y)
		print("X shape fit", X.shape)

		covs = self._compute_covariance_matrices(X, y)
		eigen_values, eigen_vectors = linalg.eigh(covs[0], covs.sum(0))
	
		ix = np.argsort(np.abs(eigen_values - 0.5))[::-1]
		eigen_vectors = eigen_vectors[:, ix]
		print("y shape", y.shape)
		self.filters_ = eigen_vectors.T
		pick_filters = self.filters_[:self.n_components]
		print("Xshape2", X.shape)
		X = X.reshape(X.shape[0], 64, 64)
		print("pick_filters shape", pick_filters.shape)
		X1 = np.asarray([np.dot(pick_filters, epoch) for epoch in X])
		print("X1 shape3", X1.shape)

		X = X1.mean(axis=2)
		print("X mean shape4", X.shape)

		return self

	def transform(self, X):
		print("X transform shape", X.shape)
		X = X.reshape(X.shape[0], 64, 64)
		pick_filters = self.filters_[:self.n_components]
		print("pick filter shape", pick_filters.shape)
		X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])
		print("X transform6", X.shape)

		# compute features (mean band power)
		X = (X ** 2).mean(axis=1)
		# X = np.log(X)	
		X -= X.mean()
		X /= X.std()
		print("X transform 7", X.shape)
		return X

	def fit_transform(self, X, y):
		print("X shape fit_transform", X.shape)
		self.fit(X, y)
		print("x fit 2 shape", X.shape)
		print("y fit 2 sahpe", y.shape)
		return self.transform(X)