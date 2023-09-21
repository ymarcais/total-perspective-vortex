import pandas as pd
import matplotlib.pyplot as plt
import mne
import os
import sklearn

#print(sklearn.__version__)

class Preprocessing:

	def __init__(self):
		pass

	#Change to capital letter the second string letter
	def capitalize_letter_at_index(self, input_string, index):
		if 0 <= index < len(input_string):
			return input_string[:index] + input_string[index].upper() + input_string[index + 1:]
		else:
			return input_string

	#- i and j will increment number of directorhy and number of file
	#- will need double loop to open directories and files
	def	edf_load(self, i, j ):
		base_url = 'physionet.org/files/eegmmidb/1.0.0/'
		directory_number = 'S' + str(i).zfill(3)
		file_number = 'S' + str(i).zfill(3) + 'R' + str(j).zfill(2) + '.edf'
		data_raw_path = os.path.join(base_url, directory_number, file_number)
		raw = mne.io.read_raw_edf(data_raw_path, preload=True)
		return raw

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

		for old_channel, new_channel_type in channel_mapping.items():
			if old_channel in info['ch_names']:
				index = info['ch_names'].index(old_channel)
				info['chs'][index]['kind'] = new_channel_type

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
	raw =  pp.rename_existing_mapping(raw)
	data = raw.get_data()
	NaN = pp.distribution_NaN(data)
	print("NaN count: ", NaN)

	filtered = pp.filering(raw, lower_passband, higher_passband)
	ica = pp.eog_artefacts()
	ica.fit(filtered)
	ica.plot_components(picks=None, ch_type=None, outlines="head", sphere='auto')


if __name__ == "__main__":
	main()