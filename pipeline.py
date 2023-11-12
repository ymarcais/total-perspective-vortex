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

'''


ecg -> electro cardiogram
ecg_projs, ecg_events = mne.preoporcessing.compute_proj_ecg(raw, n_grad=1, n_mag=1, n_egg=0, average=True)

eog-> electro oculogram
eog_projs, ecg_events = mne.preoporcessing.compute_proj_eog(raw, n_grad=1, n_mag=1, n_egg=1, average=True)

projs = ecg_projs + eog_projs
epochs.add_proj(projs)
epochs_cleaned = epochs.copy().apply_proj()


pipeline : standard + vectorize + logistic

use: score = 'roc_auc'

'''

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
	
class ICAWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=0.95, random_state=None):
        self.ica = ICA(n_components=n_components, random_state=random_state)

    def fit(self, X, y=None):
        self.ica.fit(X)
        return self

    def transform(self, X):
        return self.ica.transform(X)

class Ica_Comp(BaseEstimator, TransformerMixin):
	def __init__(self, raw=None):
		self.raw = raw
		self.ica_wrapper = ICAWrapper()

	def fit(self, X, y=None):
		if not isinstance(X, mne.io.BaseRaw):
			raise ValueError("Input must be an instance of MNE Raw, but got {}".format(type(X)))
		
		self.ica_wrapper.fit(X)
		return self

	def transform(self, X):
		return self.ica_wrapper.transform(X)



class Pipeline:

	def __init__(self, raw=None):
		self.lower_passband = 7
		self.higher_passband = 79
		self.raw=raw

	
	def set_ipochs_ica(self, raw, epochs):
		''' We want to create new epochs for ica but keeping previous events selcted after filtering
			Research shows better ICA results with l_freq = 1
			We create un new epochs subset called epochs_selection'''
		raw.filter(l_freq=1, h_freq=40)
		epochs_selction =  epochs.selction

		events, events_id = mne.events_from_annotations(raw)
		events = events[epochs_selction]

		tmin, tmax = epochs.tmin, epochs.tmax
		baseline = baseline
		epochs_ica = mne.Epochs(raw,
							event=events,
							tmin=tmin,
							tmax=tmax,
							baseline=baseline,
							preload=True
							)
		print("epochs_ica:", epochs_ica.info)
		return epochs_ica
		
	def fit_ica(self, epochs_ica):
		"n_components = 0,95 keeps 95% of the variance, and create an ica object from epochs_ica"
		n_components = 0.95
		method = 'picard'
		max_iter = 500
		fit_params =  dict(fastica_it=5)
		random_state = 42

		ica = mne.preprocessing.ICA(n_components = n_components,
									max_pca_components=300,
									method = method,
									max_iter = max_iter,
									fit_params =  fit_params,
									random_state = 42
									)
		ica.fit(epochs_ica)
		return ica
	
	def detect_ecg(self, raw, ica):
		"detect electrocadiogram articats"
		ecg_epochs = mne.preprocessing.create_ecg_epochs(raw, reject=None, baseline=(None, -0.2), tmin=-0.5, tmax=0.5)
		
		ecg_evocked = ecg_epochs.average()
		ecg_inds, ecg_scores = ica.find_bad_ecg(ecg_epochs, method='ctps')
		ecg_evocked = ecg_epochs.average()
		return ecg_inds
	
	def detect_eog(self, raw, ica):
		"detect electrocadiogram articats"
		eog_epochs = mne.preprocessing.create_eog_epochs(raw, reject=None, baseline=(None, -0.2), tmin=-0.5, tmax=0.5)
		
		eog_evocked = eog_epochs.average()
		eog_inds, ecg_scores = ica.find_bad_ecg(eog_epochs)
		eog_evocked = eog_epochs.average()
		return eog_inds
	
	def exclude_ecg_eog(self, raw, ica, epochs):
		''' Now we have discovered components related to artifacts that we want to remove 
			we apply ica to data previously generated, not to the epochs_ica'''
		ecg_inds = self.detect_eog(raw, ica)
		eog_inds = self.detect_eog(raw, ica)
		components_to_exclude = ecg_inds + eog_inds
		ica.exclude = components_to_exclude
		epochs_cleaned = ica.apply(epochs.copy())
		return epochs_cleaned









		ecg -> electro cardiogram
		ecg_projs, ecg_events = mne.preoporcessing.compute_proj_ecg(raw, n_grad=1, n_mag=1, n_egg=0, average=True)

		eog-> electro oculogram
		eog_projs, ecg_events = mne.preoporcessing.compute_proj_eog(raw, n_grad=1, n_mag=1, n_egg=1, average=True)

		projs = ecg_projs + eog_projs
		epochs.add_proj(projs)
		epochs_cleaned = epochs.copy().apply_proj()





	'''def resampling(self, sfreq=80):
		resampling_transformer = FunctionTransformer(lambda raw: self.pp.resample(raw, sfreq))
		return resampling_transformer'''
	

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
		data_ica = Ica_Comp()
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
		if not isinstance(raw, mne.io.BaseRaw):
			raise ValueError("Rooo Input must be an instance of MNE Raw, but got {}".format(type(raw)))
		#categorical_features = self.preprocessing_categorical_pipeline()
		ica_features_features = self.preprocessing_numerical_pipeline(raw)
		numerical_features = ica_features_features.fit_transform(raw)

		
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
			model.fit(raw, target)
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

