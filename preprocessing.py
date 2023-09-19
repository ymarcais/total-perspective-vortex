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
		#raw.plot()
		#plt.show()
		return raw
	
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
	filtered = pp.filering(raw, lower_passband, higher_passband)
	ica = pp.eog_artefacts()
	ica.fit(filtered)
	ica.plot_components(picks=None, ch_type=None, outlines="skirt", sphere='auto')
	


if __name__ == "__main__":
	main()