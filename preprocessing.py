import pandas as pd
import matplotlib as mplt
import mne
import os

print(mplt.__version__)


'''class Preprocessing:

	def __init__(self):

	- i and j will increment number of directorhy and number of file
	- will need double loop to open directories and files
	def edf_load(self, i, j ):

	base_url = 'physionet.org/files/eegmmidb/1.0.0/'

	directory_number = 'S' + str(i)
	file_number = 'S' + str(i).zfill(3) + str(j).zfill(2) + '.edf'
	data_raw_file = os.path.join(base_url, directory_number, file_number)
	return data_raw_file'''