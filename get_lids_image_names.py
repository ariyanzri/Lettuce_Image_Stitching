import numpy as np
import os
import pandas as pd

csv_locations_path = '/storage/ariyanzarei/image_coords_per_scan/season_10_scans'
csv_lids_locations_path = '/storage/ariyanzarei/image_coords_per_scan/season_10_lids.csv'

lid_locations = pd.read_csv(csv_lids_locations_path).T.to_dict().values()

scan_coords_csv = os.listdir(csv_locations_path)

final_list_associated = []

scans = {}

for scan in scan_coords_csv:
	scan_name = scan.replace('_coordinates.csv','')
	scan_data = pd.read_csv('{0}/{1}'.format(csv_locations_path,scan)).T.to_dict().values()

	scans[scan_name] = scan_data

for scan_name in scans:

	scan_data = scans[scan_name]
	final_list_associated[scan_name] = {}
	final_list_associated[scan_name]['path'] = 'iget -rKVPT /iplant/home/shared/terraref/ua-mac/level_1/season_10_yr_2020/stereoTop/{0}_bin2tif.tar.gz'.format(scan_name)
	final_list_associated[scan_name]['images'] = []

	for lid in lid_locations:
		
		lid_name = lid['lid_id']
		lid_loc = {'lat':float(lid['lat']),'lon':float(lid['lon'])}

		for img_data in scan_data:

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

			if lid_loc['lon']>=img_UL['lon'] and lid_loc['lon']<=img_UR['lon'] and lid_loc['lat']<=img_UL['lat'] and lid_loc['lat']>=img_LL['lat']:
				final_list_associated[scan_name]['images'].append(img_name)

for item in final_list_associated:
	print(item)