import pandas as pd
import matplotlib.pyplot as plt
import mne
import os
import sklearn

#print(sklearn.__version__)

class Preprocessing:

	def __init__(self):
		pass

	#- i and j will increment number of directorhy and number of file
	#- will need double loop to open directories and files
	def	edf_load(self, i, j ):
		base_url = 'physionet.org/files/eegmmidb/1.0.0/'
		directory_number = 'S' + str(i).zfill(3)
		file_number = 'S' + str(i).zfill(3) + 'R' + str(j).zfill(2) + '.edf'
		data_raw_path = os.path.join(base_url, directory_number, file_number)
		raw = mne.io.read_raw_edf(data_raw_path, preload=True)
		raw.info


		channel_mapping = {
			'Fc5.': 'FC5', 'Fc3.': 'FC3', 'Fc1.': 'FC1', 'Fcz.': 'FCz', 'Fc2.': 'FC2', 'Fc4.': 'FC4', 'Fc6.': 'FC6',
			'C5..': 'C5', 'C3..': 'C3', 'C1..': 'C1', 'Cz..': 'Cz', 'C2..': 'C2', 'C4..': 'C4', 'C6..': 'C6',
			'Cp5.': 'CP5', 'Cp3.': 'CP3', 'Cp1.': 'CP1', 'Cpz.': 'CPz', 'Cp2.': 'CP2', 'Cp4.': 'CP4', 'Cp6.': 'CP6',
			'Fp1.': 'Fp1', 'Fpz.': 'Fpz', 'Fp2.': 'Fp2',
			'Af7.': 'AF7', 'Af3.': 'AF3', 'Afz.': 'AFz', 'Af4.': 'AF4', 'Af8.': 'AF8',
			'F7..': 'F7', 'F5..': 'F5', 'F3..': 'F3', 'F1..': 'F1', 'Fz..': 'Fz', 'F2..': 'F2', 'F4..': 'F4', 'F6..': 'F6', 'F8..': 'F8',
			'Ft7.': 'FT7', 'Ft8.': 'FT8',
			'T7..': 'T7', 'T8..': 'T8', 'T9..': 'T9', 'T10.': 'T10',
			'Tp7.': 'TP7', 'Tp8.': 'TP8',
			'P7..': 'P7', 'P5..': 'P5', 'P3..': 'P3', 'P1..': 'P1', 'Pz..': 'Pz', 'P2..': 'P2', 'P4..': 'P4', 'P6..': 'P6', 'P8..': 'P8',
			'Po7.': 'PO7', 'Po3.': 'PO3', 'Poz.': 'POz', 'Po4.': 'PO4', 'Po8.': 'PO8',
			'O1..': 'O1', 'Oz..': 'Oz', 'O2..': 'O2',
			'Iz..': 'Iz'
		}
		n_channels = len(channel_mapping)
		ch_types = ['eeg'] * n_channels
		info = mne.create_info(list(channel_mapping.values()), 1000, ch_types)
		info = mne.pick_info(info, mne.pick_channels(info['ch_names'], include=list(channel_mapping.values())))
		for old_channel, new_channel_type in channel_mapping.items():
			if old_channel in info['ch_names']:
				index = info['ch_names'].index(old_channel)
				info['chs'][index]['kind'] = new_channel_type
		#info = mne.set_channel_types(info, mapping=channel_mapping)

		raw.rename_channels(channel_mapping)
		montage = mne.channels.make_standard_montage('standard_1020')
		raw.set_montage(montage)
		return raw
	
	def distribution_NaN(self, dataset):
		count_NaN = pd.isna(dataset).sum()
		return count_NaN
	
	#lower_passband 1Hz, higher_passband 20Hz
	def	filering(self, raw, lower_passband, higher_passband):
		filtered = raw.filter(lower_passband, higher_passband)
		#filtered.plot()
		#plt.show()
		return filtered
	
	def eog_artefacts(self):
		ica = mne.preprocessing.ICA(n_components=20, random_state=0)
		return ica
	
def main():
	i = 1
	j = 1
	lower_passband = 1
	higher_passband = 20
	pp = Preprocessing()
	try:
		raw = pp.edf_load(i, j)
	except FileNotFoundError as e:
		print(str(e))
		return
	
	data = raw.get_data()
	print("raw:", data)
	NaN = pp.distribution_NaN(data)
	print("NaN count: ", NaN)

	filtered = pp.filering(raw, lower_passband, higher_passband)
	ica = pp.eog_artefacts()
	ica.fit(filtered)
	ica.plot_components(picks=None, ch_type=None, outlines="head", sphere='auto')



if __name__ == "__main__":
	main()