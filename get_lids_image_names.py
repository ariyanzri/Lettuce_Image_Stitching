import numpy as np
import os
import pandas as pd

csv_locations_path = '/storage/ariyanzarei/image_coords_per_scan/season_10_scans'
csv_lids_locations_path = '/storage/ariyanzarei/image_coords_per_scan/season_10_lids.csv'

lid_locations = pd.read_csv(csv_lids_locations_path)

scan_coords_csv = os.listdir(csv_locations_path)

final_list_associated = []

for scan in scan_coords_csv:
	scan_name = scan.replace('_coordinates.csv','')
	scan_data = pd.read_csv('{0}/{1}'.format(csv_locations_path,scan))

	for lid_index,lid in lid_locations.iterrows():
		
		lid_name = lid['lid_id']
		lid_loc = {'lat':float(lid['lat']),'lon':float(lid['lon'])}

		# print('{0} - {1} - {2}'.format(lid_name,lid_loc['lat'],lid_loc['lon']))

		for img_index,img_data in scan_data.iterrows():

			img_name = img_data['Filename']
			UL = img_data['Upper left'].replace('"','').split(',')
			UR = img_data['Upper right'].replace('"','').split(',')
			LL = img_data['Lower left'].replace('"','').split(',')
			LR = img_data['Lower right'].replace('"','').split(',')
			C = img_data['Center'].replace('"','').split(',')

			img_UL = {'lat':float(UL[1]),'lon':float(UL[0])}
			img_UR = {'lat':float(UR[1]),'lon':float(UR[0])}
			img_LL = {'lat':float(LL[1]),'lon':float(LL[0])}
			img_LR = {'lat':float(LR[1]),'lon':float(LR[0])}
			img_C = {'lat':float(C[1]),'lon':float(C[0])}

			# print('\t {0} - {1} - {2} - {3} - {4} - {5}'.format(img_name,img_UL,img_UR,img_LL,img_LR,img_C))

			# if lid_loc['lon']>=img_UL['lon'] and lid_loc['lon']<=img_UR['lon'] and lid_loc['lat']<=img_UL['lat'] and lid_loc['lat']>=img_LL['lat']:
			# 	final_list_associated.append((scan_name,lid_name,img_name))

for item in final_list_associated:
	print(item)