import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
import os
from mne.io import concatenate_raws, read_raw_edf
from sklearn.preprocessing import StandardScaler
import sklearn
from mne.datasets import eegbci
import pywt
from mne.channels import make_standard_montage


#print(sklearn.__version__)

#Preprocessing, parsing and formating
class Preprocessing:

	def __init__(self):
		self.event_ids=dict(T0=0, T1=1, T2=2)

	#Change to capital letter the second string letter
	def capitalize_letter_at_index(self, input_string, index):
		if 0 <= index < len(input_string):
			return input_string[:index] + input_string[index].upper() + input_string[index + 1:]
		else:
			return input_string

	#- i and j will increment number of directorhy and number of file
	#- will need double loop to open directories and files
	'''def	edf_load(self, i, j ):
		base_url = 'physionet.org/files/eegmmidb/1.0.0/'
		directory_number = 'S' + str(i).zfill(3)
		file_number = 'S' + str(i).zfill(3) + 'R' + str(j).zfill(2) + '.edf'
		data_raw_path = os.path.join(base_url, directory_number, file_number)
		
		raw.annotations.onset[1:] += [0.001 * i for i in range(1, len(raw.annotations.onset))]
		events, _ = mne.events_from_annotations(raw)
		#print("events:", events)
		print("raw:", raw)
		return raw, events'''
	
	#  eegbci is a eeg dataset interface
	def	edf_load(self):
		base_url = '/mnt/nfs/homes/ymarcais/sgoinfre/total-perspective-vortex/physionet.org/files/eegmmidb/1.0.0/S001/S001R01.edf'
		runs = [[3, 4, 7, 8, 11, 12], [5, 6, 9, 10, 13, 14]]
		#subjects = list(range(1, 110))
		
		#range2 for coding
		subjects = list(range(1, 110))
		raw_list =[]
		

		for subject in subjects:
			for run in runs:
				raw_fnames = eegbci.load_data(subject, run, path="/sgoinfre/goinfre/Perso/ymarcais/data_mne")
				raw_fnames = [str(f) for f in raw_fnames]

				try:
					raw_objects = [mne.io.read_raw_edf(f, preload=True) for f in raw_fnames]
					raw_list.extend(raw_objects)
				except Exception as e:
					print(f"Error loading raw object: {e}")
					continue
				
		return raw_list	

	#Change channel mapping format
	def rename_existing_mapping(self, raw):
		channel_mapping = {}
		for channel_info in raw.info['chs']:
			ch_name = channel_info['ch_name']
			if ch_name[:2] in ['Cz','Fp', 'Fz', 'Pz', 'Oz', 'Iz']:
				kind = ch_name
			else:
				kind = self.capitalize_letter_at_index(ch_name, 1)
			kind = kind.replace('..', '').replace('.', '')
			channel_mapping[ch_name] = kind
			
		n_channels = len(channel_mapping)	
		ch_types = ['eeg'] * n_channels
		info = mne.create_info(list(channel_mapping.values()), 1000, ch_types)
		info = mne.pick_info(info, mne.pick_channels(info['ch_names'], include=list(channel_mapping.values())))
		#print("info", info)

		for old_channel, new_channel_type in channel_mapping.items():
			if old_channel in info['ch_names']:
				index = info['ch_names'].index(old_channel)
				info['chs'][index]['kind'] = new_channel_type

		raw.rename_channels(channel_mapping)
		montage = mne.channels.make_standard_montage('standard_1020')
		raw.set_montage(montage)
		return raw
	
	
	def magnitude_normaliz(self, data):
		#data = raw.get_data()
		#transposed_data = data.T

		scaler = StandardScaler()
		normalized_data = scaler.fit_transform(data)

		normalized_data = normalized_data.T
		normalized_raw = mne.io.RawArray(normalized_data, raw.info)
		return normalized_raw
	

	#check NaN
	def distribution_NaN(self, dataset):
		count_NaN = pd.isna(dataset).sum()
		return count_NaN
	
	
	#lower_passband 1Hz, higher_passband 20Hz
	def	filering(self, raw, lower_passband, higher_passband):
		filtered = raw.filter(lower_passband, higher_passband)
		return filtered
	
	'''transform data into 80 Hz freaquency domaine
	apply_function() is a method in MNE-Python used to apply a 
	given function along a specified axis of the data'''
	def resampling(self, raw):
		return raw.resample(sfreq = 80)
	

	'''#frequency fourrier transform
	def frequency_fourrier(self, raw):
		return raw.apply_function(np.fft.fft, axis = 1)'''
	def frequency_fourrier(self, data, raw):
		#data = raw.get_data()
		if data.ndim == 1:
			data = data.reshape(1, -1)

		
		data_fft = np.fft.fft(data, axis=1)
		data_fft_abs = np.abs(data_fft)
		info = mne.create_info(raw.info['ch_names'], raw.info['sfreq'], ch_types='eeg')
		raw_fft_abs = mne.io.RawArray(data_fft_abs, info)
		raw_fft_abs.set_annotations(raw.annotations)
		return raw_fft_abs
	
	
	#PSD : how the power of a signal is distributed across different frequencies.
	def psd(self, data_fft):
		fs = 80
		psd = np.abs(data_fft)**2
		psd_transp = psd.T
		print("psd shape", psd.shape)
		freq = np.fft.fftfreq(data_fft.shape[0], 1/fs)
		freq= freq[:24400]
		print("freq shape", freq.shape)
		return psd_transp, freq
	

	#select and plot psd
	def plot_psd(self, psd, freq, fft_result):
		data_shape = fft_result.get_data().shape
		num_channels = data_shape[0]
		plt.figure(figsize=(10, 5))
		for i in range(num_channels):
			if psd[i].max() >= 0.01:
				plt.scatter(freq, psd[i], label=f'Channel {i+1}', marker='o')
		plt.xlabel('Frequency (Hz)')
		plt.ylabel('Power Spectral Density (PSD)')
		plt.title('Power Spectral Density (PSD)')
		plt.legend()
		plt.grid(True)
		plt.show()
		

	def wavelet_analysis(self, raw):
		data = raw.get_data()
		wavelet = 'db1'
		coeffs = pywt.wavedec(data, wavelet)
		reconstructed_data = pywt.waverec(coeffs, wavelet, level=2)
		wavelet_raw = mne.io.RawArray(reconstructed_data, raw.info)
		wavelet_raw.set_annotations(raw.annotations)
		print("wavelet info: ", len(wavelet_raw.annotations))
		return wavelet_raw

	#get number of events
	def num_events(self, raw):
		annotations = raw.annotations
		num_events = len(annotations)
		print("Num events:", num_events)
		return num_events
	
	#mother
	def preprocessing_(self):
		lower_passband = 0
		higher_passband = 60
		results = [] 
		try:
			raw_list = self.edf_load()
		except FileNotFoundError as e:
			print(str(e))
			return
		for raw in raw_list:
			raw =  self.rename_existing_mapping(raw)
			data = raw.get_data()

			NaN = self.distribution_NaN(data)
			print("NaN count: ", NaN)
			self.num_events(raw)

			filtered= self.filering(raw, lower_passband, higher_passband)
			data1 = self.resampling(filtered)
			wavelet_raw = self.wavelet_analysis(filtered)
			raw_fft_result, data_fft = self.frequency_fourrier(data1)
			psd, freq = self.psd(data_fft)
			self.plot_psd(psd, freq, raw_fft_result)

			result = (data_fft, raw_fft_result, wavelet_raw)
			results.append(result)

			'''normalized_data_fft = self.magnitude_normaliz(data_fft)
			data = normalized_data_fft.get_data()'''

			'''ica = self.eog_artefacts()
			ica.fit(raw_fft_result)
			ica.plot_components(picks=None, ch_type='eeg', colorbar=True, outlines="head", sphere='auto')'''
		return data_fft, raw_fft_result, wavelet_raw


def main():
	i = 1
	j = 2

	pp = Preprocessing()
	raw_fft_result = pp.preprocessing_(i, j)

if __name__ == "__main__":
	main()