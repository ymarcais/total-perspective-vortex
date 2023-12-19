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

from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit, cross_validate, cross_val_score
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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
import traceback
from sklearn.decomposition import FastICA


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

		#Extract X list from epochs and convert to array
		#Extract y from event
		# len(X) = len(y)
		def get_X_y(self, epochs, events):
			X_ = []
			X = []
			y = []
			X_ = epochs.get_data()
			X = np.array(X_)
			y = events[:, 2]
			info = epochs.info
			return X, y, info

		
		#Split will be use in  cross validation
		#n_split represent the number of tine we will define different train and test samples (we dont do it one time)
		def get_n_split(self, X, y):
			n_splits = 5
			n_splits = min(5, len(X), len(y))
			return n_splits
		
		#Check size : n_split cant be < 2 and train index null
		def check_size(self, n_splits, cv, X, y, groups):
			check = False
			if n_splits < 2:
				print("Skipping fold due to insufficient splits.")
				check = True
			else:
				check = False

			for i, (train_index, test_index) in enumerate(cv.split(X, y, groups=groups)):
				if check:
					return False
				if len(train_index) == 0:
					print("Skipping fold due to no training samples.")
					check = True

				if n_splits <= len(X):
					unique_classes_train = np.unique(y[train_index])
					if len(unique_classes_train) == 1:
						print("  Warning: Only one class present in the training set.")
					return False
			return True
		
		def classifier(self, X, y,n_components):
			X_2d = X.reshape(X.shape[0], -1)
			scaler = StandardScaler()
			scaler.fit(X_2d)
			X_2d = scaler.transform(X_2d)
			clf = make_pipeline(Vectorizer(), LogisticRegression(max_iter=1000, solver='lbfgs'))
			clf.fit(X_2d, y)
			return clf
		

		def print_scores(self, clf, X, y, groups, cv):
			scoring = ['accuracy']
			print("X.shape", X.shape)
			X_2D = X.reshape(X.shape[0], -1)
			print("X_2D shape", X_2D.shape)
			y = np.array(y)
			y_flattened = y.ravel()

			scores = cross_validate(clf, X=X_2D, y=y, cv=cv, groups=groups, scoring=scoring, verbose=True)
			accuracy_mean = np.mean(scores['test_accuracy'])

			if len(np.unique(y_flattened)) > 2:  # Check if it's a multiclass problem
				clf_ovr = OneVsRestClassifier(clf)
				clf_ovr.fit(X_2D, y_flattened)
				y_scores = []
				y_scores = cross_val_predict(clf_ovr, X=X_2D, y=y_flattened, cv=5, method='predict_proba')
				y_scores_squeezed = np.squeeze(y_scores)
				y_scores = np.array(y_scores)
				y_scores = y_scores.reshape(-1, 1)
				roc_auc_mean = roc_auc_score(y_flattened, y_scores_squeezed, multi_class='ovr', average='macro')
			else:	
				# For binary classification, use regular ROC AUC
				y_scores = cross_val_score(clf, X, y_flattened, cv=cv, scoring=None)
				roc_auc_mean = roc_auc_score(y_flattened, y_scores[:, 1])
				print(f'Mean ROC AUC = {roc_auc_mean:.3f}')

			
			return accuracy_mean, roc_auc_mean, y_scores, y_flattened
		
		def num_component(self, epochs):
			epochs_data = epochs.get_data()
			n_epochs, n_channels, n_times = epochs_data.shape
			print("n_epochs", n_epochs)
			X = epochs_data.reshape((n_epochs, n_channels * n_times))

			desired_explained_variance = 0.98
			pca = PCA()
			pca.fit(X)
			cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
			num_components = np.argmax(cumulative_explained_variance >= desired_explained_variance) + 1
			if num_components < 2:
				num_components = 2
			print(f"Number of components needed to explain {desired_explained_variance * 100}% variance: {num_components}")
			return num_components


		def model(self):
			my = My_ica()
			filename_mean=[]
			all_scores_accuracy = []
			all_scores_roc_auc = []
			models = []
			score = []
			X = []
			X_reshape = []
			try:
				pp = Preprocessing()
				raw_list = pp.edf_load()
			except FileNotFoundError as e:
				print(str(e))
				return

			for raw in raw_list:
				mean_score_accuracy = []
				mean_score_roc = []
				score = []
				raw_edf_object = RawEDF(raw)
				raw_string = str(raw_edf_object)
				match = re.search(r'(S+\d+R\d+)\.edf', raw_string)
				print("raw_string", raw_string)

				if match is None:
					print("File name not matched. Skipping.")
					continue
				file_name = match.group(1)
				# apply cutoff 1hz as recommended of ICA
				raw.filter(l_freq=1, h_freq=8)

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
				annotations = raw.annotations
				unique_annotations = np.unique(annotations.description)
				event_id = {annotation: idx for idx, annotation in enumerate(unique_annotations)}
				events, event_id = mne.events_from_annotations(raw, event_id=event_id)
				class_labels = list(event_id.keys())
				print("class_labels", class_labels)

				
				if len(events) == 0:
					print("No events found. Skipping.")
					continue
				#picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
				epochs = Epochs(raw_denoised, events, event_id, tmin=0.09, tmax=0.48, baseline=(0.09, 0.48), picks='eeg', preload=True)
				n_components = self.num_component(epochs)
				print("Yn_components", n_components)
				ica, events = self.my.fast_ica(epochs, n_components=n_components)
				
				X, y, info = self.get_X_y(ica, events)
				if not np.any(X):
					print("Warning: No valid data after ICA. Skipping.")
					continue
				print("X shape", X.shape)

				fastica_epochs = my.fast_ica_sklearn(epochs, n_components, max_iterations=500, tol=1e-5)
				correlation_matrix = np.corrcoef(X, fastica_epochs)
				correlation_coefficient = correlation_matrix[0, 1]				

				X_reshape = X
					
				X_reshape = X_reshape.reshape(X.shape[0], X.shape[1] * X.shape[2])
				print("XXX X_reshape shape", X_reshape.shape)
				scalar = StandardScaler()
				scalar.fit(X_reshape)
				X_reshape = scalar.transform(X_reshape)
				X_reshape.reshape(X.shape[0], X.shape[1], X.shape[2])
				print("YYY X_reshape shape", X_reshape.shape)


				n_splits = self.get_n_split(X, y)
				split = []
				split.append(n_splits)
				cv = StratifiedGroupKFold(n_splits=n_splits)

				groups = np.arange(len(X_reshape))
				
				check = self.check_size(n_splits, cv, X_reshape, y, groups)
				if check is not check:
					continue
				
				class_scores = {}
				#for class_label, class_epochs in class_epochs_list:
				#n_components, ica_data_averaged = self.num_component(class_epochs)
				n_components = int(n_components)
				clf = self.classifier(X_reshape, y, n_components=n_components)

				accuracy, roc_auc_mean, y_scores, y_flattened = self.print_scores(clf, X_reshape, y, groups, cv)
				class_scores[file_name] = {"accuracy": accuracy}, {"roc_auc_mean": roc_auc_mean}
				print(f"accuracy: {accuracy}, roc_auc_mean: {roc_auc_mean}")
				score.append((accuracy, roc_auc_mean))
				if roc_auc_mean == -1:
					pass
				else:
					score.append((accuracy, roc_auc_mean))
	
				print("score YYY", score)
				if score:
					mean_score_accuracy = round(np.mean(score), 3)
					mean_score_roc = round(np.mean(roc_auc_mean), 3)
				else:
					mean_score_accuracy = None
				
				if mean_score_roc < 0.3:
					print("file name", file_name)
					print("WARNING LOW SCORE", mean_score_roc )
					print("n_components:", n_components)
					print("X_reshape shape", X_reshape.shape)
					print("X_reshape", X_reshape)
					print("y", y)
					print("y scores", y_scores)   ### Flatten y_scores?????
					print("y scores shape", y_scores.shape)
					print("y_flattened shape", y_flattened.shape)
					#break
			
				all_scores_accuracy.append(mean_score_accuracy)
				all_scores_roc_auc.append(mean_score_roc) 

			total_mean_score_accuracy = 0
			if accuracy:
				total_mean_score_accuracy = round(np.mean(all_scores_accuracy), 3)
			else:
				mean_score_accuracy = None
			for file_name, data in class_scores.items():
				print(f"class_scores {file_name}, accuracy: {data[0]}, roc_auc_mean: {data[1]}\n")
			total_mean_score_roc = round(np.mean(all_scores_roc_auc), 3)
			print("Mean score roc_auc", all_scores_roc_auc)
			print(f"Total_score_accuracy = {total_mean_score_accuracy}")
			print(f"Total_score_roc_auc = {total_mean_score_roc}")
			print(f"Correlation between custom and scikit-learn FastICA sources: {correlation_coefficient}")
			return True
	

def main():
	pp = MyPipeline()
	start_time = time.time()

	try:
		models = pp.model()
		end_time = time.time()

		temps = end_time - start_time
		print(f"Temps ecoule: {temps:.4f} secondes")
		#print(f"Models: {models}")

		if models:
			print("Models trained successfully.")
		else:
			print("Failed to train models.")

	except Exception as e:
		print(f"An error occurred: {e}")
		traceback.print_exc()

if __name__ == "__main__":
	main()

