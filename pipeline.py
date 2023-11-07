import numpy as np
import os
import mne
import matplotlib.pyplot as plt
import pywt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import FunctionTransformer

from mne.datasets import eegbci
from mne.channels import _standard_montage_utils

from preprocessing import Preprocessing
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from my_ica import My_ica
from treatment_pipeline import Treatment_pipeline
from mne.filter import filter_data
from mne.preprocessing import ICA


from sklearn.base import BaseEstimator, TransformerMixin
import mne

class MNEFilterTransformer(BaseEstimator, TransformerMixin):
	def __init__(self, sfreq=None, l_freq=7, h_freq=40):
		self.lower_passband = l_freq
		self.higher_passband = h_freq
		self.sfreq = sfreq

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		if isinstance(X, mne.io.BaseRaw):  # Check if X is an MNE Raw object
			data = X.get_data()  # Get the raw data from the Raw object
			sfreq = X.info['sfreq']  # Get the sampling frequency from the Raw object
		else:
			data = X
			sfreq = self.sfreq

		filtered_data = mne.filter.filter_data(data, sfreq=sfreq, l_freq=self.lower_passband, h_freq=self.higher_passband)

		return filtered_data

class Wavelet(BaseEstimator, TransformerMixin):
	def __init__(self, raw=None) :
		self.raw = raw

	def fit(self, X, y=None):
		return self
	
	
	def transform(self, X):
		if isinstance(X, mne.io.BaseRaw):  # Check if X is an MNE Raw object
			data = X.get_data()  # Get the raw data from the Raw object
		else:
			data = X
		
		wavelet = 'db1'
		level = 2
		coeffs = pywt.wavedec(data, wavelet)
		reconstructed_data = pywt.waverec(coeffs[:level + 1], wavelet)
		wavelet_raw = mne.io.RawArray(reconstructed_data, self.raw.info)
		wavelet_raw.set_annotations(self.raw.annotations)

		return wavelet_raw
	
class Fourrier_Frequency_Transformation(BaseEstimator, TransformerMixin):
	def __init__(self, raw=None) :
		self.raw = raw

	def fit(self, X, y=None):
		return self
	
	
	def transform(self, X):
		if isinstance(X, mne.io.BaseRaw):  # Check if X is an MNE Raw object
			data = X.get_data()  # Get the raw data from the Raw object
		else:
			data = X

		data_fft = np.fft.fft(data, axis=1)
		return data_fft
	
	
class Ica_Comp(BaseEstimator, TransformerMixin):
	def __init__(self, raw=None) :
		self.raw = raw


	def fit(self, X, y=None):
		return self
	
	
	def transform(self, X):
		data_fft_abs = np.abs(X)
		raw_fft_result = mne.io.RawArray(data_fft_abs, self.raw.info)
		raw_fft_result.set_annotations(self.raw.annotations)
		random_seed = 42
		ica = ICA(n_components=0.95, random_state=random_seed)
		ica_result = ica.fit(raw_fft_result)
		n_components = ica.n_components_
		return ica_result

class Pipeline:

	def __init__(self) -> None:
		self.lower_passband = 7
		self.higher_passband = 79


	def resampling(self, sfreq=80):
		'''Function to resample the raw data'''
		resampling_transformer = FunctionTransformer(lambda raw: self.pp.resample(raw, sfreq))
		return resampling_transformer
	

	def preprocessing_numerical_pipeline(self, raw):
		''' Create a nunerical pipeline
				frequence normalization
				filter lower && uper bands
				frequency fourrier transformation
				wavelet analysis
				get ica components reduction with 95% of variance
				LDA'''
		pp = Preprocessing()
		tp = Treatment_pipeline()
		sfreq = raw.info['sfreq']
		
		filter = MNEFilterTransformer(sfreq=sfreq, l_freq =1, h_freq =40)
		scaler = StandardScaler()
		wavelet = Wavelet(raw=raw)
		data_fft = Fourrier_Frequency_Transformation()
		data_ica = Ica_Comp(raw=raw)
		#lda_result = LDA(self)
		
		
		numerical_pipeline = make_pipeline(	filter,
											scaler,
											wavelet,
											data_fft,
											data_ica
											)
		
		return numerical_pipeline


	def preprocessing_categorical_pipeline(self, raw):
		''' Create a pipeline on cathegorical data:
				rename mapping
				Check inputs
				OneHotEcnoder gives converts categorical to digital matrix'''
		
'''make a class for mapping, this is not a make_pipeline'''
		mapping = make_pipeline(Preprocessing().rename_existing_mapping(raw))
		impute = SimpleImputer(strategy='most_frequent')
		categorical_pipeline = make_pipeline(	mapping,
									   			impute,
									   		)
		return categorical_pipeline

	
	def preprocessor_(self, raw):
		''' preprocessor is a transformer using to pipelines:
				numerical pipelines
				categorical pipelines'''
		

		numerical_features = self.preprocessing_numerical_pipeline(raw)
		categorical_features = self.preprocessing_categorical_pipeline(raw)

		return numerical_features, categorical_features
	
	
	def model(self):
		try:
			pp = Preprocessing()
			raw_list = pp.edf_load()
			event = pp.num_events(raw_list[0])
			print("event:", event)

		except FileNotFoundError as e:
			print(str(e))
			return
		
		models = []
		for raw in raw_list:
			numerical_features, categorical_features = self.preprocessor_(raw)
			model = make_pipeline(numerical_features, categorical_features, SGDClassifier)
			model.fit(raw.get_data(), event)
			models.append(model)
		return model
	

def main():
	pp = Pipeline()
	models = pp.model()

	if models:
		print("Models trained successfully.")
	else:
		print("Failed to train models.")

if __name__ == "__main__":
	main()

