import matplotlib.pyplot as plt
import mne
from preprocessing import Preprocessing
from sklearn.decomposition import PCA
from mne.preprocessing import ICA
from mne.decoding import CSP

class Treatment_pipeline():

	def __init__(self) -> None:
		self.pp = Preprocessing()

	#principal component analysis
	def pca(self, i, j, num_pca_components):
		data_fft, raw_fft_result, wavelet_raw= self.pp.preprocessing_(i, j)
		psd, freq = self.pp.psd(data_fft)
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
		return pca_result, raw_fft_result, wavelet_raw
	
	#get statistically independent underlying signals 
	def eog_artefacts(self):
		ica = mne.preprocessing.ICA(n_components=18, random_state=0)
		return ica
	
	#fit ica from instance of Raw raw_fft_result
	def ica_comp(self, raw_fft_result):
			random_seed = 42
			ica = ICA(n_components=0.95, random_state=random_seed)
			ica_result = ica.fit(raw_fft_result)
			n_components = ica.n_components_
			print("n_components:", n_components)
			return ica_result, n_components

	#plot brain heads activity with a number of n_components 
	def ica_plot(self, ica_result, n_components):
		ica_result.plot_components(
			picks=None, 
			ch_type='eeg', 
			colorbar=True, 
			outlines="head", 
			sphere='auto',
			title= f'ICA reduction {n_components} components',
								)
	#def csp_comp(self, n_components=2, )
	#Mother
	def treatment_pipeline(self,i, j, num_pca_components):
		pca_result, raw_fft_result, wavelet_raw = self.pca(i, j, num_pca_components)

		ica_result, n_components = self.ica_comp(wavelet_raw)
		self.ica_plot(ica_result, n_components)


def main():
	
	i = 4
	j = 4
	tp = Treatment_pipeline()
	num_pca_components = 2
	tp.treatment_pipeline(i, j, num_pca_components)

if __name__ == "__main__":
	main()


