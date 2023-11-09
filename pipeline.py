import numpy as np
import os
import mne
import matplotlib.pyplot as plt
import pywt
from mne.channels import DigMontage

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import FunctionTransformer

from mne.datasets import eegbci
from mne.channels import _standard_montage_utils
from mne.channels import make_standard_montage

from preprocessing import Preprocessing
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from my_ica import My_ica
from treatment_pipeline import Treatment_pipeline
from mne.filter import filter_data
from mne.preprocessing import ICA
from sklearn.pipeline import Pipeline
from mne import create_info



from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_union
import mne

class MNEFilterTransformer(BaseEstimator, TransformerMixin):

	def __init__(self, sfreq=None, l_freq=7, h_freq=40):
		self.lower_passband = l_freq
		self.higher_passband = h_freq

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		if isinstance(X, mne.io.BaseRaw):  # Check if X is an MNE Raw object
			data = X.get_data()  # Get the raw data from the Raw object
			sfreq = X.info['sfreq']  # Get the sampling frequency from the Raw object
		else:
			data = X
			sfreq = 160

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

		if isinstance(X, mne.io.BaseRaw):  # Check if X is an MNE Raw object
			data = X.get_data()  # Get the raw data from the Raw object
		else:
			data = X
		data_fft_abs = np.abs(data)
		raw_fft_result = mne.io.RawArray(data_fft_abs, self.raw.info)
		raw_fft_result.set_annotations(self.raw.annotations)
		random_seed = 42
		ica = ICA(n_components=0.95, random_state=random_seed)
		ica_result = ica.fit(raw_fft_result)
		n_components = ica.n_components_
		return ica_result
	

class Rename_existing_mapping(BaseEstimator, TransformerMixin):

	def __init__(self, raw=None) :
		self.raw = raw


	def fit(self, X, y=None):
		return self
	
	
	def transform(self, raw):
		channel_mapping={}
		'''expected_channel_64 = [
								'Fp1', 'Fpz', 'Fp2',
								'F7', 'F3', 'Fz', 'F4', 'F8',
								'FC5', 'FC1', 'FC2', 'FC6',
								'M1', 'T7', 'C3', 'Cz', 'C4', 'T8', 'M2',
								'CP5', 'CP1', 'CP2', 'CP6',
								'P7', 'P3', 'Pz', 'P4', 'P8',
								'PO9', 'O1', 'Oz', 'O2', 'PO10',
								'AF7', 'AF3', 'AF4', 'AF8',
								'F5', 'F1', 'F2', 'F6',
								'FC3', 'FCz', 'FC4',
								'C5', 'C1', 'C2', 'C6',
								'CP3', 'CPz', 'CP4',
								'P5', 'P1', 'P2', 'P6',
								'PO7', 'P09', 'Oz', 'O2', 'PO10'
							]'''
		
		for channel_info in raw.info['chs']:
			ch_name = channel_info['ch_name']
			if ch_name in raw.info['chs']:
				kind = ch_name
			else:
				kind = ch_name.rstrip('.')
				if kind == 'Fpz':
					pass
				else:
					kind = kind.upper()
				if kind[:2] == 'FP':
					kind = 'Fp' + kind[2:].lower()
				if kind.endswith('Z'):
					kind = kind[:-1] + 'z'
			
			channel_mapping[ch_name] = kind
		n_channels = len(channel_mapping)
		ch_types = ['eeg'] * n_channels
		info = mne.create_info(list(channel_mapping.values()), 1000, ch_types)
		info = mne.pick_info(info, mne.pick_channels(info['ch_names'], include=list(channel_mapping.values())))

		raw.rename_channels(channel_mapping)
		montage = mne.channels.make_standard_montage('standard_1020')
		raw.set_montage(montage)
		return raw


class Pipeline:

	def __init__(self, raw=None):
		self.lower_passband = 7
		self.higher_passband = 79
		self.raw=raw


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
		if isinstance(raw, mne.io.BaseRaw):
			print("The object is an instance of RawEDF")
		else:
			print("The object is not an instance of RawEDF")
		sfreq = raw.info['sfreq']
		
		filter = MNEFilterTransformer(sfreq=sfreq, l_freq =1, h_freq =40)
		scaler = StandardScaler()
		wavelet = Wavelet(raw)
		data_fft = Fourrier_Frequency_Transformation()
		data_ica = Ica_Comp(raw)
		#lda_result = LDA(self)
		
		
		numerical_pipeline = make_pipeline(	filter,
											scaler,
											wavelet,
											data_fft,
											data_ica
											)
		return numerical_pipeline


	def preprocessing_categorical_pipeline(self):
		''' Create a pipeline on cathegorical data:
				rename mapping
				Check inputs
				OneHotEcnoder gives converts categorical to digital matrix'''

		mapping = Rename_existing_mapping()
		impute = SimpleImputer(strategy='most_frequent')
		categorical_pipeline = make_pipeline(	mapping,
									   			impute,
									   		)
		return categorical_pipeline

	
	def preprocessor_(self, raw):
		''' preprocessor is a transformer using to pipelines:
				numerical pipelines
				categorical pipelines'''
		
		#categorical_features = self.preprocessing_categorical_pipeline()
		numerical_features = self.preprocessing_numerical_pipeline(raw)
		
		#preporcessor = make_union(categorical_features, numerical_features)

		return numerical_features
	
	
	def model(self):
				
		try:
			pp = Preprocessing()
			raw_list = pp.edf_load()
						
		except FileNotFoundError as e:
			print(str(e))
			return
		
		models = []
		event = np.array([])
		for raw in raw_list:
			events, event_id = mne.events_from_annotations(raw)
			event_id_values = list(event_id.values())
			events, event_id = mne.events_from_annotations(raw)
			target = events[:, 2] 
			sfreq_value = raw.info['sfreq'] 
			preporcessor = self.preprocessor_(raw)
			model = make_pipeline(preporcessor, SGDClassifier())
			ica_data = raw.get_data()
			print("ica_data", ica_data)
			model.fit(ica_data, target)
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

