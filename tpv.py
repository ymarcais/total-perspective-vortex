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
				print("XXXX raw denoised info", raw_denoised.info)

				events, event_id = mne.events_from_annotations(raw, event_id=dict(T0=0, T1=1, T2=2))

				print("event = :", events)

				if len(events) == 0:
					print("No events found. Skipping.")
					continue
				epochs = mne.Epochs(raw_denoised, events, tmin=-0.25, tmax=0.5, baseline=(None, 0), picks='eeg')

				# Optionally, you can still use ICA for additional artifact removal
				ica = ICA(n_components=0.95, method='picard', max_iter=500)
				ica.fit(epochs)
				ica.exclude = []  # Adjust based on your needs

				# Apply ICA to further clean the data
				epochs_cleaned = ica.apply(epochs.copy())
				self.classifier(epochs_cleaned)
				models.append(epochs_cleaned)

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

