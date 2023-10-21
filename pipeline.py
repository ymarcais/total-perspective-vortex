import numpy as np
import os
import mne 
import matplotlib.pyplot as plt

from my_ica import Ica
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean, std
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import SGDClassifier
from joblib import dump

from mne.io import concatenate_raws
from mne.datasets import eegbci
from mne import events_from_annotations
from mne.channels import _standard_montage_utils

from preprocessing import Preprocessing
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from my_ica import My_ica
from treatment_pipeline import Treatment_pipeline



class Pipeline:

	def __init__(self) -> None:
		self.lower_passband = 7
		self.higher_passband = 80

	def preprocessing_numerical_pipeline(self):
		''' Create a nunerical pipeline
				frequence normalization
				filter lower && uper bands
				frequency fourrier transformation
				wavelet analysis
				get ica components reduction with 95% of variance
				LDA'''
		
		numerical_pipeline = make_pipeline(Preprocessing().magnitude_normaliz(), 
										   filter(self.lower_passband, self.higher_passband), 
										   mne.resample(sfreq = 80),
										   Preprocessing().frequency_fourrier(),
										   Preprocessing().wavelet_analysis(),
										   Treatment_pipeline().ica_comp(),
										   LDA(),
										   )
		return numerical_pipeline


	def preprocessing_categorical_pipeline(self):
		''' Create a pipeline on cathegorical data:
				rename mapping
				Check inputs
				OneHotEcnoder gives converts categorical to digital matrix'''
		
		categorical_pipeline = make_pipeline(Preprocessing().rename_existing_mapping(),
									   		SimpleImputer(strategy='most_frequent'),
									   		OneHotEncoder)
		return categorical_pipeline

		
	def preprocessor_(self, numerical_features, categorical_features):
		''' preprocessor is a transformer using to pipelines:
				numerical pipelines
				categorical pipelines'''
		numerical_pipeline = self.preprocessing_numerical_pipeline()
		categorical_pipeline = self.preprocessing_categorical_pipeline()
		preprocessor = make_column_transformer((numerical_pipeline, numerical_features),
											(categorical_pipeline, categorical_features))
		return preprocessor
	
	
	def model(self, numerical_features, categorical_features):
		X, y = Preprocessing().edf_load()
		preprocessor = self.preprocessor_(numerical_features, categorical_features)
		model = make_pipeline(preprocessor, SGDClassifier)
		model.fit(X, y)





