import numpy as np
import os
import re
from io import StringIO
import sys

from CSP import CSP


import mne
import matplotlib.pyplot as plt
import pywt
from mne.channels import DigMontage
from mne.preprocessing import compute_proj_ecg, compute_proj_eog
from mne import Epochs, pick_types, events_from_annotations

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit, cross_val_score
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
from mne import set_config
from mne.decoding import SlidingEstimator, cross_val_multiscore

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

class RawEDF:
    def __init__(self, filename):
        self.filename = filename


    def __str__(self):
        return f"RawEDF | {self.filename}), data loaded"
		
class MyPipeline:
		def __init__(self):
			pass

		def classifier(self, epochs_cleaned, num_components):
			all_scores = []
			n_splits = 5
			scoring = 'roc_auc'
			epochs_cleaned = epochs_cleaned.copy().pick_types(meg=False, eeg=True, eog=False, exclude='bads')
			X = epochs_cleaned.get_data()
			print("X shape", X.shape)
			n_splits = min(n_splits, len(X))
			y = epochs_cleaned.events[:, 2]
			n_splits = min(n_splits, len(y))
			class_counts = np.bincount(y)
			n_splits = min(n_splits, len(class_counts) - 1)
			groups = np.arange(len(epochs_cleaned))
			cv = StratifiedGroupKFold(n_splits=n_splits)
			csp = CSP(num_components)

			for i, (train_index, test_index) in enumerate(cv.split(X, y, groups=groups)):
				print(f"Debugging fold {i + 1}/{n_splits}...")
				if n_splits < 2:
					print("Skipping fold due to insufficient splits.")
					return -1, -1
				if len(train_index) == 0:
					print("Skipping fold due to no training samples.")
					return -1, -1

				# rest of the code for cross-validation

				if len(X) >= n_splits:

					# Check if train_index is not empty before accessing groups
					if len(train_index) > 0:
						print(f"  xxxTrain: index={train_index}")
						print(f"         xxxgroup={groups[train_index]}")
					else:
						print("  xxxTrain: No training samples in this fold.")

					print(f"  xxxTest:  index={test_index}")
					print(f"         xxxgroup={groups[test_index]}")
					unique_classes_train = np.unique(y[train_index])
					if len(unique_classes_train) == 1:
						print("  Warning: Only one class present in the training set.")
					continue 
			print("epochs_cleaned.info", epochs_cleaned.info)
									
			clf = make_pipeline(Scaler(epochs_cleaned.info), Vectorizer(), LogisticRegression('l2'))
			print("XX shape", X.shape)
			print("Data type of X:", type(X))

			scores = cross_val_score(clf, X=X, y=y, cv=cv, groups=groups, scoring=scoring)
			all_scores.extend(scores)
			if all_scores:
				roc_auc_mean = round(np.mean(all_scores), 3)
				roc_auc_std = round(np.std(all_scores), 3)


			
			print(f"CV scores: {scores}")
			print(f'Mean ROC AUC = {roc_auc_mean:.3f} (SD = {roc_auc_std:.3f})')
			
			return clf, roc_auc_mean

		def model(self):
			filename_mean=[]
			try:
				pp = Preprocessing()
				raw_list = pp.edf_load()
			except FileNotFoundError as e:
				print(str(e))
				return

			models = []
			score = []
			

			for raw in raw_list:
				raw_edf_object = RawEDF(raw)
				raw_string = str(raw_edf_object)
				match = re.search(r'(S+\d+R\d+)\.edf', raw_string)
				file_name = match.group(1)
				raw.filter(l_freq=0.1, h_freq=40)

				# Assuming your EEG channels are in ch_names
				ch_names = raw.ch_names
				data, times = raw[:, :]

				# Apply wavelet denoising to each EEG channel

				denoised_data = np.zeros_like(data)
				for i, ch_name in enumerate(ch_names):
					coeffs = pywt.wavedec(data[i, :], 'db1', level=4)  # Adjust the wavelet and level as needed
					threshold = np.std(coeffs[-1]) * 2  # Adjust the threshold as needed
					coeffs[1:] = (pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:])
					denoised_data[i, :] = pywt.waverec(coeffs, 'db1')
					#denoised_data[i, :] = pp.magnitude_normaliz(denoised_data[i, :])
				
				# Create MNE Raw object from the denoised data
				raw_denoised = mne.io.RawArray(denoised_data, raw.info)
				event_id = dict(T1=1, T2=2)
				events, event_id = mne.events_from_annotations(raw, event_id=event_id)
				#event_id_binary = {'T1': 0, 'T2': 1}
				#events, event_id = mne.events_from_annotations(raw, event_id=event_id)

				if len(events) == 0:
					print("No events found. Skipping.")
					continue
				picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
				epochs = Epochs(raw_denoised, events, event_id, tmin=0.09, tmax=0.46, baseline=(0.09, 0.1), picks='eeg', preload=True)
				

				# Optionally, you can still use ICA for additional artifact removal
				ica = ICA(n_components=0.98, method='picard', verbose=1)
				original_stdout = sys.stdout
				sys.stdout = StringIO()
				ica.fit(epochs)
				captured_output = sys.stdout.getvalue()
				sys.stdout = original_stdout
				print("XXXcapture output", captured_output)
				matches = re.search(r"Selecting by explained variance: (\d+) components", captured_output)
				num_components = int(matches.group(1))
				print("XXXn_components", matches)

				
				ica.exclude = []  # Adjust based on your needs
				
				

				# Apply ICA to further clean the data
				epochs_cleaned = ica.apply(epochs.copy())
				_, roc_auc_mean = self.classifier(epochs_cleaned, num_components)
				filename_mean.append({"file_name": file_name, "roc_auc_mean": roc_auc_mean})
				
				if roc_auc_mean == -1:
					pass
				else:
					score.append(roc_auc_mean)
				models.append(epochs_cleaned)
				del raw
			
			mean_score = round(np.mean(score), 3) * 100
			for data in filename_mean:
				print(f"File name: {data['file_name']}, roc_auc_mean: {data['roc_auc_mean']}\n")

			print(f"Final mean score {mean_score}%")
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

