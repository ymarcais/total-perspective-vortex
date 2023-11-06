import numpy as np
import os
import mne
import matplotlib.pyplot as plt

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


class Filtering:
	'''create a class filtering to create a transformer that use 
	parameters lower_passband, higher_passband'''
	def __init__(self, lower_passband, higher_passband):
		self.lower_passband = lower_passband
		self.higher_passband = higher_passband
		pp = Preprocessing()

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		# Your filtering logic here, using self.lower_passband and self.higher_passband
		filtered_data = self.pp.filetring(X, self.lower_passband, self.higher_passband)
		return filtered_data

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

		
		filtered = FunctionTransformer(Filtering(self.lower_passband, self.higher_passband).transform)
		resemp = self.resampling(filtered)
		normalized = FunctionTransformer(pp.magnitude_normaliz(filtered))
		wavelet = FunctionTransformer(pp.wavelet_analysis(filtered))
		raw_fft_result, data_fft = FunctionTransformer(pp.frequency_fourrier(resemp))
		_, data_ica = FunctionTransformer(tp.ica_comp(raw_fft_result))
		lda_result = LDA(data_fft)
		
		numerical_pipeline = make_pipeline(	filtered,
									 		resemp,
									 		normalized,
									 		wavelet,											
											raw_fft_result, data_fft,
											data_ica,
											lda_result
											)
		return numerical_pipeline


	def preprocessing_categorical_pipeline(self, raw):
		''' Create a pipeline on cathegorical data:
				rename mapping
				Check inputs
				OneHotEcnoder gives converts categorical to digital matrix'''
		
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





