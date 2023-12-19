import numpy as np
import os
import re
from io import StringIO
import sys
import warnings

from CSP import CSP
from sklearn.decomposition import PCA
import time


import mne
import matplotlib.pyplot as plt
import pywt
from mne.channels import DigMontage
from mne.preprocessing import compute_proj_ecg, compute_proj_eog
from mne import Epochs, pick_types, events_from_annotations

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit, cross_validate
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
			self.my = My_ica()

		
		def get_X_y(self, epochs_cleaned):
			epochs_cleaned = epochs_cleaned.copy().pick_types(meg=False, eeg=True, eog=False, exclude='bads')
			X = epochs_cleaned.get_data()
			y = epochs_cleaned.events[:, 2]
			return X, y


		def get_n_split(self, X, y):
			n_splits = 5
			n_splits = min(5, len(X), len(y), len(np.bincount(y)) - 1)
			return n_splits
		

		def check_size(self, n_splits, train_index):
			if n_splits < 2:
				print("Skipping fold due to insufficient splits.")
				return -1, -1
			if len(train_index) == 0:
				print("Skipping fold due to no training samples.")
				return -1, -1
			else:
				return True
		
		def classifier(self, epochs_cleaned):
			groups = np.arange(len(epochs_cleaned))
			check = False
				
			X, y = self.get_X_y(epochs_cleaned)
			n_splits = self.get_n_split(X, y)
			cv = StratifiedGroupKFold(n_splits=n_splits)

			for i, (train_index, test_index) in enumerate(cv.split(X, y, groups=groups)):
				check = self.check_size(n_splits, train_index)
				if check:
					continue

				if n_splits <= len(X):
					unique_classes_train = np.unique(y[train_index])
					if len(unique_classes_train) == 1:
						print("  Warning: Only one class present in the training set.")
					continue 
			print("epochs_cleaned.info", epochs_cleaned.info)
									
			clf = make_pipeline(Scaler(epochs_cleaned.info), Vectorizer(), LogisticRegression('l2'))
			return clf, X, y, groups, cv
		

		def print_scores(self, clf, X, y, groups, cv):
			scoring = 'roc_auc', 'accuracy'
			all_scores = []
			scores = cross_validate(clf, X=X, y=y, cv=cv, groups=groups, scoring=scoring)
			#all_scores.extend(scores)
			roc_auc_mean = round(np.mean(scores['test_roc_auc']), 3)
			print("1")
			print("roc_auc_mean", roc_auc_mean)
			accuracy_mean = round(np.mean(scores['test_accuracy']), 3)
			print("accuracy_mean", accuracy_mean)
			#roc_auc_std = round(np.std(all_scores), 3)
			
			print(f'Mean ROC AUC = {roc_auc_mean:.3f} (Accuracy = {accuracy_mean:.3f})')
			
			return roc_auc_mean, accuracy_mean
		
		def num_component(self, epochs):
			epochs_data = epochs.get_data()
			n_epochs, n_channels, n_times = epochs_data.shape
			X = epochs_data.reshape((n_epochs, n_channels * n_times))

			desired_explained_variance = 0.20
			pca = PCA()
			pca.fit(X)
			cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
			num_components = np.argmax(cumulative_explained_variance >= desired_explained_variance) + 1
			if num_components < 2:
				num_components = 2
			print(f"Number of components needed to explain {desired_explained_variance * 100}% variance: {num_components}")
			return num_components, X


		def model(self):
			filename_mean=[]
			models = []
			score = []
			try:
				pp = Preprocessing()
				raw_list = pp.edf_load()
			except FileNotFoundError as e:
				print(str(e))
				return

			for raw in raw_list:
				raw_edf_object = RawEDF(raw)
				raw_string = str(raw_edf_object)
				match = re.search(r'(S+\d+R\d+)\.edf', raw_string)
				print("match :", match)
				if match is None:
					print("File name not matched. Skipping.")
					continue
				file_name = match.group(1)

				raw.filter(h_freq=8, l_freq=0.1)

				ch_names = raw.ch_names
				data, _ = raw[:, :]

				# Applying wavelet denoising to each EEG channel

				denoised_data = np.zeros_like(data)
				for i, ch_name in enumerate(ch_names):
					coeffs = pywt.wavedec(data[i, :], 'db1', level=4)
					threshold = np.std(coeffs[-1]) * 2
					coeffs[1:] = (pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:])
					denoised_data[i, :] = pywt.waverec(coeffs, 'db1')
				
				# Create MNE Raw object from the denoised data
				raw_denoised = mne.io.RawArray(denoised_data, raw.info)
				event_id = dict(T1=1, T2=2)
				events, event_id = mne.events_from_annotations(raw, event_id=event_id)

				if len(events) == 0:
					print("No events found. Skipping.")
					continue
				picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
				epochs = Epochs(raw_denoised, events, event_id, tmin=0.09, tmax=0.48, baseline=(0.09, 0.48), picks='eeg', preload=True)
				n_components, X = self.num_component(epochs)
				print("YYY n_components:", n_components)
				#ica = ICA(n_components=n_components, method='picard', verbose=1)
				ica = self.my.fast_ica(X=X, n_components=n_components, max_iterations=500)

				# Optionally, you can still use ICA for additional artifact removal
				#noise_cov = mne.compute_covariance(epochs, tmin=0, tmax=None, method='auto')
	
				original_stdout = sys.stdout
				sys.stdout = StringIO()
				ica.fit(raw)
				ica.apply(raw)
	
				
				# Apply ICA to further clean the data
				epochs_cleaned = ica.apply(epochs.copy())
				#epochs_cleaned.apply_baseline((0.10, 0.48))
				sys.stdout = original_stdout

				epochs_cleaned.apply_baseline((0.10, 0.48))
				ica.exclude = []
				if not np.any(epochs_cleaned._data):
					print("Warning: No valid data after ICA. Skipping.")
					continue

				clf, X, y, groups, cv = self.classifier(epochs_cleaned)
				roc_auc_mean, accuracy = self.print_scores(clf, X, y, groups, cv)
				filename_mean.append({"file_name": file_name, "roc_auc_mean": roc_auc_mean, "accuracy": accuracy})
				
				if roc_auc_mean == -1:
					pass
				else:
					score.append((roc_auc_mean, accuracy))
				models.append(epochs_cleaned)
				del raw
			roc_auc_means = [item[0] for item in score]
			mean_score_roc = round(np.mean(roc_auc_means	), 3) * 100
			accuracy_means = [item[1] for item in score]
			mean_score_accuracy = round(np.mean(accuracy_means), 3) * 100
			for data in filename_mean:
				print(f"File name: {data['file_name']}, roc_auc_mean: {data['roc_auc_mean']}, accuracy:{data['accuracy']}\n")


			print(f"Final mean score roc {mean_score_roc:.3f}%, accuracy {mean_score_accuracy:.3f}%")
			return models
	

def main():
	pp = MyPipeline()
	start_time = time.time()
	models = pp.model()
	end_time = time.time()

	temps = end_time - start_time
	print(f"Temps ecoule: {temps:.4f} secondes")
	if models:
		print("Models trained successfully.")
	else:
		print("Failed to train models.")

if __name__ == "__main__":
	main()

