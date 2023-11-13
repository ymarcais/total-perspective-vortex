import numpy as np
import os
import mne
import matplotlib.pyplot as plt
import pywt
from mne.channels import DigMontage
from mne.preprocessing import compute_proj_ecg, compute_proj_eog

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import FunctionTransformer

from mne.datasets import eegbci
from mne.channels import _standard_montage_utils

from preprocessing import Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from my_ica import My_ica
from treatment_pipeline import Treatment_pipeline
from mne.preprocessing import ICA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from mne.decoding import Scaler,Vectorizer

from sklearn.base import BaseEstimator, TransformerMixin
import mne

'''


ecg -> electro cardiogram
ecg_projs, ecg_events = mne.preporcessing.compute_proj_ecg(raw, n_grad=1, n_mag=1, n_egg=0, average=True)

eog-> electro oculogram
eog_projs, ecg_events = mne.preporcessing.compute_proj_eog(raw, n_grad=1, n_mag=1, n_egg=1, average=True)

projs = ecg_projs + eog_projs
epochs.add_proj(projs)
epochs_cleaned = epochs.copy().apply_proj()


pipeline : standard + vectorize + logistic

use: score = 'roc_auc'

'''
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

class MyPipeline:

	def __init__(self, raw=None):
		self.lower_passband = 7
		self.higher_passband = 79
		self.raw=raw

	def raw_filtered(self, raw):
		raw.filter(self.lower_passband, self.higher_passband)
		return raw

	def epochs(self, raw):
		tmin = -0.25
		tmax = 0.5
		baseline = (None, 0)
		events, event_id = mne.events_from_annotations(raw)

		epochs = mne.Epochs(raw,
					  		events=events,
							event_id=event_id,
							tmin=tmin,
							tmax=tmax,
							baseline=baseline)
		return epochs
	
	def epochs_ica(self, raw, epochs):
		''' We want to create new epochs for ica but keeping previous events selcted after filtering
			Research shows better ICA results with l_freq = 1
			We create un new epochs subset called epochs_selection'''
		raw.filter(l_freq=1, h_freq=40)
		epochs_selection =  epochs.selection

		events, events_id = mne.events_from_annotations(raw)
		events = events[epochs_selection]

		tmin, tmax = epochs.tmin, epochs.tmax
		baseline = (None, 0)
		epochs_ica = mne.Epochs(raw,
							events=events,
							tmin=tmin,
							tmax=tmax,
							baseline=baseline,
							preload=True,
							picks='eeg' 
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
									method = method,
									max_iter = max_iter,
									fit_params =  fit_params,
									random_state = 42,
									)
		ica.fit(epochs_ica)
		return ica
	
	'''def detect_ecg(self, raw, ica):
		"detect electrocadiogram articats"
		#ecg_epochs = mne.preprocessing.create_ecg_epochs(raw, reject=None, baseline=(None, -0.2), tmin=-0.5, tmax=0.5)
		ecg_epochs = mne.preprocessing.create_ecg_epochs(raw, picks='eeg')
		
		ecg_evocked = ecg_epochs.average()
		ecg_inds, ecg_scores = ica.find_bad_ecg(ecg_epochs, method='ctps')
		ecg_evocked = ecg_epochs.average()
		ecg_evoked = ecg_epochs.average()
		return ecg_inds'''
	def detect_ecg(self, raw, ica):
		#ecg_epochs = mne.preprocessing.create_ecg_epochs(raw, picks='eeg')
		ecg_epochs = mne.preprocessing.create_ecg_epochs(raw,  picks='eeg', reject=None, baseline=(None, -0.2), tmin=-0.5, tmax=0.5)
		ecg_evoked = ecg_epochs.average()
		ecg_inds, ecg_scores = ica.find_bads_ecg(ecg_epochs, method='ctps')
		return ecg_inds
	
	
	def detect_eog(self, raw, ica):
		"detect electrocadiogram articats"
		eog_epochs = mne.preprocessing.create_eog_epochs(raw, picks='eeg')
		#eog_epochs = mne.preprocessing.create_eog_epochs(raw, reject=None, baseline=(None, -0.2), tmin=-0.5, tmax=0.5)
		
		eog_evocked = eog_epochs.average()
		eog_inds, eog_scores = ica.find_bad_eog(eog_epochs)
		eog_evocked = eog_epochs.average()
		return eog_inds
	

	def exclude_ecg_eog(self, raw, ica, epochs):
		''' Now we have discovered components related to artifacts that we want to remove 
			we apply ica to data previously generated, not to the epochs_ica'''
		ecg_inds = self.detect_eog(raw, ica)
		eog_inds = self.detect_eog(raw, ica)
		components_to_exclude = ecg_inds 
		ica.exclude = components_to_exclude
		epochs_cleaned = ica.apply(epochs.copy())
		return epochs_cleaned
	
	def classifier(self, epochs_cleaned):
		n_splits = 5
		scoring = 'roc_auc'
		epochs_cleaned = epochs_cleaned.copy().pick_types(meg=False, eeg=True, eog=True)
		X = epochs_cleaned.get_data()
		y = epochs_cleaned.events[:, 2]
		cv = StratifiedGroupKFold(n_splits=n_splits)
		clf = make_pipeline(Scaler(epochs_cleaned.info), Vectorizer(), LogisticRegression())
		scores = cross_val_score(clf, X=X, y=y, scoring=scoring)
		roc_auc_mean = round(np.mean(scores), 3)
		roc_auc_std = round(np.std(scores), 3)

		print(f"CV scores: {scores}")
		print(f'Mean ROC AUC = {roc_auc_mean:.3f} (SD = {roc_auc_std:.3f})')
		return clf
	
	def model(self):
				
		try:
			pp = Preprocessing()
			raw_list = pp.edf_load()
						
		except FileNotFoundError as e:
			print(str(e))
			return
		
		models = []
		#event = np.array([])
		for raw in raw_list:
			raw = self.raw_filtered(raw)
			epochs =  self.epochs(raw)
			epochs_ica = self.epochs_ica(raw, epochs)
			ica = self.fit_ica(epochs_ica)
			epochs_cleaned = self.exclude_ecg_eog(raw, ica, epochs)
			models = self.classifier(epochs_cleaned)
		return models
	

def main():
	pp = MyPipeline()
	models = pp.model()

	if models:
		print("Models trained successfully.")
	else:
		print("Failed to train models.")

if __name__ == "__main__":
	main()

