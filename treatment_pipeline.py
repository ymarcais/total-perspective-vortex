import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
import os
from preprocessing import Preprocessing
#from preprocessing import raw_fft_result
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sklearn

class Treatment_pipeline():

	def __init__(self) -> None:
		self.pp = Preprocessing()

	#principal component analysis
	def cpa(self, i, j, num_pca_components):
		data_fftt = self.pp.preprocessing_(i, j)
		psd, freq = self.pp.psd(data_fftt)
		'''data_fft = raw_fft_result.get_data()
		data_fft_abs = np.abs(data_fft)'''
		flattened_psd = psd.reshape(psd.shape[0], -1)
		pca = PCA(n_components=num_pca_components)
		pca_result = pca.fit_transform(flattened_psd)

		plt.scatter(pca_result[:, 0], pca_result[:, 1])
		plt.xlabel('Principal Component 1')
		plt.ylabel('Principal Component 2')
		plt.title('PCA after PSD')
		plt.show()
		return pca_result


def main():
	
	i = 6
	j = 1
	tp = Treatment_pipeline()
	num_pca_components = 2
	tp.cpa(i, j, num_pca_components)

	


if __name__ == "__main__":
	main()


