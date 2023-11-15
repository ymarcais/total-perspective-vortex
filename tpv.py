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

class Rename_existing_mapping(BaseEstimator, TransformerMixin):

	def __init__(self, raw=None) :
		self.raw = raw


	def fit(self, X, y=None):
		return self
	
	
	def transform(self, raw):
		channel_mapping={}
				
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
		info = mne.create_info(list(channel_mapping.values()), 100, ch_types)
		info = mne.pick_info(info, mne.pick_channels(info['ch_names'], include=list(channel_mapping.values())))

		raw.rename_channels(channel_mapping)
		montage = mne.channels.make_standard_montage('standard_1020')
		raw.set_montage(montage)
		return raw

class MyPipeline:

		def classifier(self, epochs_cleaned):
			n_splits = 5
			scoring = 'roc_auc'
			epochs_cleaned = epochs_cleaned.copy().pick_types(meg=False, eeg=True, eog=False, exclude='bads')
			X = epochs_cleaned.get_data()
			y = epochs_cleaned.events[:, 2]
			cv = StratifiedGroupKFold(n_splits=n_splits)
			clf = make_pipeline(Scaler(epochs_cleaned.info), Vectorizer(), LogisticRegression())
			scores = cross_val_score(clf, X=X, y=y, scoring=scoring)
			roc_auc_mean = round(np.mean(scores), 3)
			roc_auc_std = round(np.std(scores), 3)

			print(f"CV scores: {scores}")
			print(f'Mean ROC AUC = {roc_auc_mean:.3f} (SD = {roc_auc_std:.3f})')
			return clf, roc_auc_mean

		def model(self):
			try:
				pp = Preprocessing()
				raw_list = pp.edf_load()
			except FileNotFoundError as e:
				print(str(e))
				return

			models = []
			score = []

			for raw in raw_list:
				sfreq = raw.info['sfreq']
				raw.filter(l_freq=1, h_freq=40)
				print("raw annotations ===", raw.annotations)

				# Assuming your EEG channels are in ch_names
				ch_names = raw.ch_names
				data, times = raw[:, :]

				# Apply wavelet denoising to each EEG channel
				denoised_data = np.zeros_like(data)
				for i, ch_name in enumerate(ch_names):
					print("ch_name", ch_name)
					coeffs = pywt.wavedec(data[i, :], 'db1', level=4)  # Adjust the wavelet and level as needed
					threshold = np.std(coeffs[-1]) * 2  # Adjust the threshold as needed
					coeffs[1:] = (pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:])
					denoised_data[i, :] = pywt.waverec(coeffs, 'db1')

				# Create MNE Raw object from the denoised data
				raw_denoised = mne.io.RawArray(denoised_data, raw.info)
				events, event_id = mne.events_from_annotations(raw, event_id=dict(T0=0, T1=1, T2=2))
				event_id_binary = {'T1': 0, 'T2': 1}
				events, event_id = mne.events_from_annotations(raw, event_id=event_id_binary)

				if len(events) == 0:
					print("No events found. Skipping.")
					continue
				epochs = mne.Epochs(raw_denoised, events, tmin=-0.25, tmax=0.5, baseline=(None, 0), picks='eeg', preload=True)

				# Optionally, you can still use ICA for additional artifact removal
				ica = ICA(n_components=0.95, method='picard', max_iter=500)
				ica.fit(epochs)
				ica.exclude = []  # Adjust based on your needs

				# Apply ICA to further clean the data
				epochs_cleaned = ica.apply(epochs.copy())
				_, roc_auc_mean = self.classifier(epochs_cleaned)
				score.append(roc_auc_mean)
				print("score", score)
				models.append(epochs_cleaned)
				del raw
				#raw.close()
			mean_score = np.mean(score)
			print("mean score", mean_score)
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

