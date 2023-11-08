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

	def capitalize_letter_at_index(self, input_string, index):
		if 0 <= index < len(input_string):
			return input_string[:index] + input_string[index].upper() + input_string[index + 1:]
		else:
			return input_string

	def fit(self, X, y=None):
		return self
	
	def transform(self, raw):
		channel_mapping = {}
		for channel_info in raw.info['chs']:
			ch_name = channel_info['ch_name']
			if ch_name[:2] in ['Cz','Fp', 'Fz', 'Pz', 'Oz', 'Iz']:
				kind = ch_name
			else:
				kind = self.capitalize_letter_at_index(ch_name, 1)
			kind = kind.replace('..', '').replace('.', '')
		channel_mapping[ch_name] = kind
		print("channel mapping", channel_mapping)
		n_channels = len(channel_mapping)
		print("n_channels", n_channels)
		ch_types = ['eeg'] * n_channels
		print("ch_types", ch_types)
		info = mne.create_info(list(channel_mapping.values()), 1000, ch_types)
		info = mne.pick_info(info, mne.pick_channels(info['ch_names'], include=list(channel_mapping.values())))

		for old_channel, new_channel_type in channel_mapping.items():
			if old_channel in info['ch_names']:
				index = info['ch_names'].index(old_channel)
				info['chs'][index]['kind'] = new_channel_type

		raw.rename_channels(channel_mapping)
		montage = make_standard_montage('standard_1020')
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
		
		categorical_features = self.preprocessing_categorical_pipeline()
		numerical_features = self.preprocessing_numerical_pipeline(raw)
		
		preporcessor = make_union(categorical_features, numerical_features)

		return preporcessor
	
	
	def model(self):
		
		try:
			pp = Preprocessing()
			raw_list = pp.edf_load()
			print("raw_list", raw_list)
			#event = pp.num_events(raw_list[0][0])
			
		except FileNotFoundError as e:
			print(str(e))
			return
		
		models = []
		for raw in raw_list:
			print("raw_list", raw)
			event = pp.num_events(raw)
			event = np.array(event).reshape(-1)
			print("event shape", event.shape)
			sfreq_value = raw.info['sfreq'] 
			print("XXX sfreq_value", sfreq_value)
			preporcessor = self.preprocessor_(raw)
			
			model = make_pipeline(preporcessor, SGDClassifier())
			print("sizeof channels:", raw.info['ch_names'])
			model.fit(raw, event)
			print("toto")
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

