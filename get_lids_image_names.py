import numpy as np
import os
import pandas as pd

csv_locations_path = '/storage/ariyanzarei/image_coords_per_scan/season_10_scans'
csv_lids_locations_path = '/storage/ariyanzarei/image_coords_per_scan/season_10_lids.csv'

lid_locations = pd.read_csv(csv_lids_locations_path)

scan_coords_csv = os.listdir(csv_locations_path)

for scan in scan_coords_csv:
	scan_data = pd.read_csv('{0}/{1}'.format(csv_locations_path,scan))

	for lid_index,lid in lid_locations.iterrows():
		
		lid_name = lid['lid_id']
		lid_lat = lid['lat']
		lid_long = lid['lon']

		# print('{0} - {1} - {2}'.format(lid_name,lid_lat,lid_long))

		for img_index,img_data in scan_data.iterrows():

			img_name = img_data['Filename']
			img_UL = img_data['Upper left'].replace('"','').split(',')
			img_UR = img_data['Upper right'].replace('"','').split(',')
			img_LL = img_data['Lower left'].replace('"','').split(',')
			img_LR = img_data['Lower right'].replace('"','').split(',')
			img_C = img_data['Center'].replace('"','').split(',')

			print('\t {0} - {1} - {2} - {3} - {4} - {5}'.format(img_name,img_UL,img_UR,img_LL,img_LR,img_C))