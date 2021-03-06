import datetime

def initialize_settings(scan_date,config_file,local_address):
	global method,no_of_cores_to_use,no_of_cores_to_use_max,SCALE,PATCH_SIZE,LID_SIZE_AT_SCALE,PATCH_SIZE_GPS,\
	GPS_TO_IMAGE_RATIO,HEIGHT_RATIO_FOR_ROW_SEPARATION,PERCENTAGE_OF_GOOD_MATCHES,MINIMUM_PERCENTAGE_OF_INLIERS,\
	MINIMUM_NUMBER_OF_MATCHES,RANSAC_MAX_ITER,RANSAC_ERROR_THRESHOLD,PERCENTAGE_NEXT_NEIGHBOR_FOR_MATCHES,\
	OVERLAP_DISCARD_RATIO,TRANSFORMATION_SCALE_DISCARD_THRESHOLD,TRANSFORMATION_ANGLE_DISCARD_THRESHOLD,\
	LETTUCE_AREA_THRESHOLD,CONTOUR_MATCHING_MIN_MATCH,ORTHO_SCALE,REDUCTION_FACTOR,OPEN_MORPH_LID_SIZE,\
	CLOSE_MORPH_LID_SIZE,FFT_PARALLEL_CORES_TO_USE,use_camera,override_sifts,\
	patch_folder,SIFT_folder,lid_file,coordinates_file,CORRECTED_coordinates_file,plot_npy_file,\
	row_save_path,field_image_path,lettuce_heads_coordinates_file,correction_log_file,inside_radius_lettuce_matching_threshold,\
	number_of_rows_in_groups,groups_to_use,patches_to_use,scan_date_stng,is_single_group,is_flir,\
	lid_detection_method,temp_lid_image_address,circle_error,lid_search_surrounding_patch_number,\
	TRANSFORMATION_ERR_STD,GPS_ERROR_STD,LID_ERR_STD,lines,Height_Scale,LID_SIZE,lid_detection_model_path,\
	save_coords_on_csv,save_new_tiffs,new_tiffs_path

	with open(config_file,'r') as f:
		lines = f.read().split('\n')

		size_init = int(lines[24].split(':')[1].split(',')[0]),int(lines[24].split(':')[1].split(',')[1])
		LID_SIZE = int(lines[25].split(':')[1].split(',')[0]),int(lines[25].split(':')[1].split(',')[1])
		
		method = lines[0].split(':')[1]
		no_of_cores_to_use = int(lines[1].split(':')[1])
		no_of_cores_to_use_max = int(lines[2].split(':')[1])
		SCALE = float(lines[3].split(':')[1])
		PATCH_SIZE = (int(size_init[0]*SCALE),int(size_init[1]*SCALE))
		Height_Scale = (1.7999999997186933e-05,2.0499999997980467e-05)
		LID_SIZE_AT_SCALE = (-1,-1)
		PATCH_SIZE_GPS = (-1,-1)
		GPS_TO_IMAGE_RATIO = (PATCH_SIZE_GPS[0]/PATCH_SIZE[1],PATCH_SIZE_GPS[1]/PATCH_SIZE[0])
		HEIGHT_RATIO_FOR_ROW_SEPARATION = float(lines[4].split(':')[1])
		PERCENTAGE_OF_GOOD_MATCHES = float(lines[5].split(':')[1])
		MINIMUM_PERCENTAGE_OF_INLIERS = float(lines[6].split(':')[1])
		MINIMUM_NUMBER_OF_MATCHES = int(lines[7].split(':')[1])
		RANSAC_MAX_ITER = int(lines[8].split(':')[1])
		RANSAC_ERROR_THRESHOLD = int(lines[9].split(':')[1])
		PERCENTAGE_NEXT_NEIGHBOR_FOR_MATCHES = float(lines[10].split(':')[1])
		OVERLAP_DISCARD_RATIO = float(lines[11].split(':')[1])
		TRANSFORMATION_SCALE_DISCARD_THRESHOLD = float(lines[12].split(':')[1])
		TRANSFORMATION_ANGLE_DISCARD_THRESHOLD = float(lines[13].split(':')[1])
		LETTUCE_AREA_THRESHOLD = int(lines[14].split(':')[1])
		CONTOUR_MATCHING_MIN_MATCH = int(lines[15].split(':')[1])
		ORTHO_SCALE = float(lines[16].split(':')[1])
		REDUCTION_FACTOR = ORTHO_SCALE/SCALE
		OPEN_MORPH_LID_SIZE = int(lines[17].split(':')[1])
		CLOSE_MORPH_LID_SIZE = int(lines[18].split(':')[1])
		

		FFT_PARALLEL_CORES_TO_USE = int(lines[19].split(':')[1])
		use_camera = lines[20].split(':')[1]
		override_sifts = (True if lines[21].split(':')[1] == 'true' or lines[21].split(':')[1] == 'True' else False)
		inside_radius_lettuce_matching_threshold = 200*SCALE
		number_of_rows_in_groups = int(lines[22].split(':')[1])
		groups_to_use = slice(0,None)
		patches_to_use = slice(0,None)
		is_single_group = (True if lines[23].split(':')[1] == 'true' or lines[23].split(':')[1] == 'True' else False)
		is_flir = False
		lid_detection_method = lines[26].split(':')[1]
		circle_error = int(lines[27].split(':')[1])
		lid_search_surrounding_patch_number = int(lines[28].split(':')[1])
		TRANSFORMATION_ERR_STD = float(lines[29].split(':')[1].split(',')[0]),float(lines[29].split(':')[1].split(',')[1])
		GPS_ERROR_STD = float(lines[30].split(':')[1].split(',')[0]),float(lines[30].split(':')[1].split(',')[1])
		LID_ERR_STD = float(lines[31].split(':')[1])

		lid_detection_model_path = lines[32].split(':')[1]

		save_coords_on_csv = (True if lines[33].split(':')[1] == 'true' or lines[33].split(':')[1] == 'True' else False)
		save_new_tiffs = (True if lines[34].split(':')[1] == 'true' or lines[34].split(':')[1] == 'True' else False)
		new_tiffs_path = '{0}/{1}-rgb/output_tiffs'.format(local_address,scan_date)

		temp_lid_image_address = '{0}/{1}-rgb/lid_temp.png'.format(local_address,scan_date)

		patch_folder = '{0}/{1}-rgb/bin2tif_out'.format(local_address,scan_date)
		SIFT_folder = '{0}/{1}-rgb/SIFT'.format(local_address,scan_date)
		lid_file = '{0}/{1}-rgb/lids.txt'.format(local_address,scan_date)
		coordinates_file = '{0}/{1}-rgb/{2}_coordinates.csv'.format(local_address,scan_date,scan_date)
		CORRECTED_coordinates_file = '{0}/{1}-rgb/{2}_coordinates_CORRECTED.csv'.format(local_address,scan_date,scan_date)
		plot_npy_file = '{0}/{1}-rgb/plt.npy'.format(local_address,scan_date)
		row_save_path = '{0}/{1}-rgb/rows'.format(local_address,scan_date)
		field_image_path = '{0}/{1}-rgb'.format(local_address,scan_date)
		correction_log_file = '{0}/{1}-rgb/logs/log_{2}_at_{3}.csv'.format(local_address,scan_date,method,datetime.datetime.now().strftime("%d-%m-%y_%H:%M"))
		lettuce_heads_coordinates_file = '{0}/{1}-rgb/season10_ind_lettuce_2020-05-27.csv'.format(local_address,scan_date)
		scan_date_stng = scan_date
		lettuce_coords = None


def initialize_settings_HPC(scan_date,config_file,destination,lid_add,bin2tif_address,repository_address):
	global method,no_of_cores_to_use,no_of_cores_to_use_max,SCALE,PATCH_SIZE,LID_SIZE_AT_SCALE,PATCH_SIZE_GPS,\
	GPS_TO_IMAGE_RATIO,HEIGHT_RATIO_FOR_ROW_SEPARATION,PERCENTAGE_OF_GOOD_MATCHES,MINIMUM_PERCENTAGE_OF_INLIERS,\
	MINIMUM_NUMBER_OF_MATCHES,RANSAC_MAX_ITER,RANSAC_ERROR_THRESHOLD,PERCENTAGE_NEXT_NEIGHBOR_FOR_MATCHES,\
	OVERLAP_DISCARD_RATIO,TRANSFORMATION_SCALE_DISCARD_THRESHOLD,TRANSFORMATION_ANGLE_DISCARD_THRESHOLD,\
	LETTUCE_AREA_THRESHOLD,CONTOUR_MATCHING_MIN_MATCH,ORTHO_SCALE,REDUCTION_FACTOR,OPEN_MORPH_LID_SIZE,\
	CLOSE_MORPH_LID_SIZE,FFT_PARALLEL_CORES_TO_USE,use_camera,override_sifts,\
	patch_folder,SIFT_folder,lid_file,coordinates_file,CORRECTED_coordinates_file,plot_npy_file,\
	field_image_path,lettuce_heads_coordinates_file,correction_log_file,inside_radius_lettuce_matching_threshold,\
	number_of_rows_in_groups,groups_to_use,patches_to_use,scan_date_stng,is_single_group,is_flir,\
	lid_detection_method,temp_lid_image_address,circle_error,lid_search_surrounding_patch_number,\
	TRANSFORMATION_ERR_STD,GPS_ERROR_STD,LID_ERR_STD,lines,Height_Scale,LID_SIZE,lid_detection_model_path,\
	save_coords_on_csv,save_new_tiffs,new_tiffs_path

	with open(config_file,'r') as f:
		lines = f.read().split('\n')

		size_init = int(lines[24].split(':')[1].split(',')[0]),int(lines[24].split(':')[1].split(',')[1])
		LID_SIZE = int(lines[25].split(':')[1].split(',')[0]),int(lines[25].split(':')[1].split(',')[1])
		
		method = lines[0].split(':')[1]
		no_of_cores_to_use = int(lines[1].split(':')[1])
		no_of_cores_to_use_max = int(lines[2].split(':')[1])
		SCALE = float(lines[3].split(':')[1])
		PATCH_SIZE = (int(size_init[0]*SCALE),int(size_init[1]*SCALE))
		Height_Scale = (1.7999999997186933e-05,2.0499999997980467e-05)
		LID_SIZE_AT_SCALE = (-1,-1)
		PATCH_SIZE_GPS = (-1,-1)
		GPS_TO_IMAGE_RATIO = (PATCH_SIZE_GPS[0]/PATCH_SIZE[1],PATCH_SIZE_GPS[1]/PATCH_SIZE[0])
		HEIGHT_RATIO_FOR_ROW_SEPARATION = float(lines[4].split(':')[1])
		PERCENTAGE_OF_GOOD_MATCHES = float(lines[5].split(':')[1])
		MINIMUM_PERCENTAGE_OF_INLIERS = float(lines[6].split(':')[1])
		MINIMUM_NUMBER_OF_MATCHES = int(lines[7].split(':')[1])
		RANSAC_MAX_ITER = int(lines[8].split(':')[1])
		RANSAC_ERROR_THRESHOLD = int(lines[9].split(':')[1])
		PERCENTAGE_NEXT_NEIGHBOR_FOR_MATCHES = float(lines[10].split(':')[1])
		OVERLAP_DISCARD_RATIO = float(lines[11].split(':')[1])
		TRANSFORMATION_SCALE_DISCARD_THRESHOLD = float(lines[12].split(':')[1])
		TRANSFORMATION_ANGLE_DISCARD_THRESHOLD = float(lines[13].split(':')[1])
		LETTUCE_AREA_THRESHOLD = int(lines[14].split(':')[1])
		CONTOUR_MATCHING_MIN_MATCH = int(lines[15].split(':')[1])
		ORTHO_SCALE = float(lines[16].split(':')[1])
		REDUCTION_FACTOR = ORTHO_SCALE/SCALE
		OPEN_MORPH_LID_SIZE = int(lines[17].split(':')[1])
		CLOSE_MORPH_LID_SIZE = int(lines[18].split(':')[1])
		

		FFT_PARALLEL_CORES_TO_USE = int(lines[19].split(':')[1])
		use_camera = lines[20].split(':')[1]
		override_sifts = (True if lines[21].split(':')[1] == 'true' or lines[21].split(':')[1] == 'True' else False)
		inside_radius_lettuce_matching_threshold = 200*SCALE
		number_of_rows_in_groups = int(lines[22].split(':')[1])
		groups_to_use = slice(0,None)
		patches_to_use = slice(0,None)
		is_single_group = (True if lines[23].split(':')[1] == 'true' or lines[23].split(':')[1] == 'True' else False)
		is_flir = False
		lid_detection_method = lines[26].split(':')[1]
		circle_error = int(lines[27].split(':')[1])
		lid_search_surrounding_patch_number = int(lines[28].split(':')[1])
		TRANSFORMATION_ERR_STD = float(lines[29].split(':')[1].split(',')[0]),float(lines[29].split(':')[1].split(',')[1])
		GPS_ERROR_STD = float(lines[30].split(':')[1].split(',')[0]),float(lines[30].split(':')[1].split(',')[1])
		LID_ERR_STD = float(lines[31].split(':')[1])

		lid_detection_model_path = lines[32].split(':')[1]

		save_coords_on_csv = (True if lines[33].split(':')[1] == 'true' or lines[33].split(':')[1] == 'True' else False)
		save_new_tiffs = (True if lines[34].split(':')[1] == 'true' or lines[34].split(':')[1] == 'True' else False)
		new_tiffs_path = '{0}/{1}/output_tiffs'.format(destination,scan_date)

		temp_lid_image_address = '{0}/lid_temp.png'.format(repository_address)

		patch_folder = '{0}'.format(bin2tif_address)
		SIFT_folder = '{0}/{1}/SIFT'.format(destination,scan_date)
		lid_file = '{0}'.format(lid_add)
		CORRECTED_coordinates_file = '{0}/{1}/{2}_coordinates_CORRECTED.csv'.format(destination,scan_date,scan_date)
		plot_npy_file = '{0}/{1}/plt.npy'.format(destination,scan_date)
		field_image_path = '{0}/{1}'.format(destination,scan_date)
		correction_log_file = '{0}/{1}/logs/log_{2}_at_{3}.csv'.format(destination,scan_date,method,datetime.datetime.now().strftime("%d-%m-%y_%H:%M"))
		scan_date_stng = scan_date
		lettuce_coords = None


def initialize_settings_test(scan_date,config_file,local_address,rows_n,patch_n):
	global method,no_of_cores_to_use,no_of_cores_to_use_max,SCALE,PATCH_SIZE,LID_SIZE_AT_SCALE,PATCH_SIZE_GPS,\
	GPS_TO_IMAGE_RATIO,HEIGHT_RATIO_FOR_ROW_SEPARATION,PERCENTAGE_OF_GOOD_MATCHES,MINIMUM_PERCENTAGE_OF_INLIERS,\
	MINIMUM_NUMBER_OF_MATCHES,RANSAC_MAX_ITER,RANSAC_ERROR_THRESHOLD,PERCENTAGE_NEXT_NEIGHBOR_FOR_MATCHES,\
	OVERLAP_DISCARD_RATIO,TRANSFORMATION_SCALE_DISCARD_THRESHOLD,TRANSFORMATION_ANGLE_DISCARD_THRESHOLD,\
	LETTUCE_AREA_THRESHOLD,CONTOUR_MATCHING_MIN_MATCH,ORTHO_SCALE,REDUCTION_FACTOR,OPEN_MORPH_LID_SIZE,\
	CLOSE_MORPH_LID_SIZE,FFT_PARALLEL_CORES_TO_USE,use_camera,override_sifts,\
	patch_folder,SIFT_folder,lid_file,coordinates_file,CORRECTED_coordinates_file,plot_npy_file,\
	row_save_path,field_image_path,lettuce_heads_coordinates_file,correction_log_file,inside_radius_lettuce_matching_threshold,\
	number_of_rows_in_groups,groups_to_use,patches_to_use,scan_date_stng,is_single_group,is_flir,\
	lid_detection_method,temp_lid_image_address,circle_error,lid_search_surrounding_patch_number,\
	TRANSFORMATION_ERR_STD,GPS_ERROR_STD,LID_ERR_STD,lines,Height_Scale,LID_SIZE,lid_detection_model_path,\
	save_coords_on_csv,save_new_tiffs,new_tiffs_path

	with open(config_file,'r') as f:
		lines = f.read().split('\n')

		size_init = int(lines[24].split(':')[1].split(',')[0]),int(lines[24].split(':')[1].split(',')[1])
		LID_SIZE = int(lines[25].split(':')[1].split(',')[0]),int(lines[25].split(':')[1].split(',')[1])
		
		method = lines[0].split(':')[1]
		no_of_cores_to_use = int(lines[1].split(':')[1])
		no_of_cores_to_use_max = int(lines[2].split(':')[1])
		SCALE = float(lines[3].split(':')[1])
		PATCH_SIZE = (int(size_init[0]*SCALE),int(size_init[1]*SCALE))
		Height_Scale = (1.7999999997186933e-05,2.0499999997980467e-05)
		LID_SIZE_AT_SCALE = (-1,-1)
		PATCH_SIZE_GPS = (-1,-1)
		GPS_TO_IMAGE_RATIO = (PATCH_SIZE_GPS[0]/PATCH_SIZE[1],PATCH_SIZE_GPS[1]/PATCH_SIZE[0])
		HEIGHT_RATIO_FOR_ROW_SEPARATION = float(lines[4].split(':')[1])
		PERCENTAGE_OF_GOOD_MATCHES = float(lines[5].split(':')[1])
		MINIMUM_PERCENTAGE_OF_INLIERS = float(lines[6].split(':')[1])
		MINIMUM_NUMBER_OF_MATCHES = int(lines[7].split(':')[1])
		RANSAC_MAX_ITER = int(lines[8].split(':')[1])
		RANSAC_ERROR_THRESHOLD = int(lines[9].split(':')[1])
		PERCENTAGE_NEXT_NEIGHBOR_FOR_MATCHES = float(lines[10].split(':')[1])
		OVERLAP_DISCARD_RATIO = float(lines[11].split(':')[1])
		TRANSFORMATION_SCALE_DISCARD_THRESHOLD = float(lines[12].split(':')[1])
		TRANSFORMATION_ANGLE_DISCARD_THRESHOLD = float(lines[13].split(':')[1])
		LETTUCE_AREA_THRESHOLD = int(lines[14].split(':')[1])
		CONTOUR_MATCHING_MIN_MATCH = int(lines[15].split(':')[1])
		ORTHO_SCALE = float(lines[16].split(':')[1])
		REDUCTION_FACTOR = ORTHO_SCALE/SCALE
		OPEN_MORPH_LID_SIZE = int(lines[17].split(':')[1])
		CLOSE_MORPH_LID_SIZE = int(lines[18].split(':')[1])
		

		FFT_PARALLEL_CORES_TO_USE = int(lines[19].split(':')[1])
		use_camera = lines[20].split(':')[1]
		override_sifts = (True if lines[21].split(':')[1] == 'true' or lines[21].split(':')[1] == 'True' else False)
		inside_radius_lettuce_matching_threshold = 200*SCALE
		number_of_rows_in_groups = int(lines[22].split(':')[1])
		groups_to_use = slice(0,rows_n)
		patches_to_use = slice(0,patch_n)
		is_single_group = (True if lines[23].split(':')[1] == 'true' or lines[23].split(':')[1] == 'True' else False)
		is_flir = False
		lid_detection_method = lines[26].split(':')[1]
		circle_error = int(lines[27].split(':')[1])
		lid_search_surrounding_patch_number = int(lines[28].split(':')[1])
		TRANSFORMATION_ERR_STD = float(lines[29].split(':')[1].split(',')[0]),float(lines[29].split(':')[1].split(',')[1])
		GPS_ERROR_STD = float(lines[30].split(':')[1].split(',')[0]),float(lines[30].split(':')[1].split(',')[1])
		LID_ERR_STD = float(lines[31].split(':')[1])

		lid_detection_model_path = lines[32].split(':')[1]

		save_coords_on_csv = (True if lines[33].split(':')[1] == 'true' or lines[33].split(':')[1] == 'True' else False)
		save_new_tiffs = (True if lines[34].split(':')[1] == 'true' or lines[34].split(':')[1] == 'True' else False)
		new_tiffs_path = '{0}/{1}-rgb/output_tiffs'.format(local_address,scan_date)

		temp_lid_image_address = '{0}/{1}-rgb/lid_temp.png'.format(local_address,scan_date)
		

		patch_folder = '{0}/{1}-rgb/bin2tif_out'.format(local_address,scan_date)
		SIFT_folder = '{0}/{1}-rgb/SIFT'.format(local_address,scan_date)
		lid_file = '{0}/{1}-rgb/lids.txt'.format(local_address,scan_date)
		coordinates_file = '{0}/{1}-rgb/{2}_coordinates.csv'.format(local_address,scan_date,scan_date)
		CORRECTED_coordinates_file = '{0}/{1}-rgb/{2}_coordinates_CORRECTED.csv'.format(local_address,scan_date,scan_date)
		plot_npy_file = '{0}/{1}-rgb/plt.npy'.format(local_address,scan_date)
		row_save_path = '{0}/{1}-rgb/rows'.format(local_address,scan_date)
		field_image_path = '{0}/{1}-rgb'.format(local_address,scan_date)
		correction_log_file = '{0}/{1}-rgb/logs/log_{2}_at_{3}.csv'.format(local_address,scan_date,method,datetime.datetime.now().strftime("%d-%m-%y_%H:%M"))
		lettuce_heads_coordinates_file = '{0}/{1}-rgb/season10_ind_lettuce_2020-05-27.csv'.format(local_address,scan_date)
		scan_date_stng = scan_date
		lettuce_coords = None

def initialize_settings_FLIR(scan_date,config_file,destination,lid_add,bin2tif_address,repository_address):
	global method,no_of_cores_to_use,no_of_cores_to_use_max,SCALE,PATCH_SIZE,LID_SIZE_AT_SCALE,PATCH_SIZE_GPS,\
	GPS_TO_IMAGE_RATIO,HEIGHT_RATIO_FOR_ROW_SEPARATION,PERCENTAGE_OF_GOOD_MATCHES,MINIMUM_PERCENTAGE_OF_INLIERS,\
	MINIMUM_NUMBER_OF_MATCHES,RANSAC_MAX_ITER,RANSAC_ERROR_THRESHOLD,PERCENTAGE_NEXT_NEIGHBOR_FOR_MATCHES,\
	OVERLAP_DISCARD_RATIO,TRANSFORMATION_SCALE_DISCARD_THRESHOLD,TRANSFORMATION_ANGLE_DISCARD_THRESHOLD,\
	LETTUCE_AREA_THRESHOLD,CONTOUR_MATCHING_MIN_MATCH,ORTHO_SCALE,REDUCTION_FACTOR,OPEN_MORPH_LID_SIZE,\
	CLOSE_MORPH_LID_SIZE,FFT_PARALLEL_CORES_TO_USE,use_camera,override_sifts,\
	patch_folder,SIFT_folder,lid_file,coordinates_file,CORRECTED_coordinates_file,plot_npy_file,\
	field_image_path,lettuce_heads_coordinates_file,correction_log_file,inside_radius_lettuce_matching_threshold,\
	number_of_rows_in_groups,groups_to_use,patches_to_use,scan_date_stng,is_single_group,is_flir,\
	lid_detection_method,temp_lid_image_address,circle_error,lid_search_surrounding_patch_number,\
	TRANSFORMATION_ERR_STD,GPS_ERROR_STD,LID_ERR_STD,lines,Height_Scale,LID_SIZE,lid_detection_model_path,\
	save_coords_on_csv,save_new_tiffs,new_tiffs_path

	with open(config_file,'r') as f:
		lines = f.read().split('\n')

		size_init = int(lines[24].split(':')[1].split(',')[0]),int(lines[24].split(':')[1].split(',')[1])
		LID_SIZE = int(lines[25].split(':')[1].split(',')[0]),int(lines[25].split(':')[1].split(',')[1])
		
		method = lines[0].split(':')[1]
		no_of_cores_to_use = int(lines[1].split(':')[1])
		no_of_cores_to_use_max = int(lines[2].split(':')[1])
		SCALE = float(lines[3].split(':')[1])
		PATCH_SIZE = (int(size_init[0]*SCALE),int(size_init[1]*SCALE))
		Height_Scale = (1.7999999997186933e-05,2.0499999997980467e-05)
		LID_SIZE_AT_SCALE = (-1,-1)
		PATCH_SIZE_GPS = (-1,-1)
		GPS_TO_IMAGE_RATIO = (PATCH_SIZE_GPS[0]/PATCH_SIZE[1],PATCH_SIZE_GPS[1]/PATCH_SIZE[0])
		HEIGHT_RATIO_FOR_ROW_SEPARATION = float(lines[4].split(':')[1])
		PERCENTAGE_OF_GOOD_MATCHES = float(lines[5].split(':')[1])
		MINIMUM_PERCENTAGE_OF_INLIERS = float(lines[6].split(':')[1])
		MINIMUM_NUMBER_OF_MATCHES = int(lines[7].split(':')[1])
		RANSAC_MAX_ITER = int(lines[8].split(':')[1])
		RANSAC_ERROR_THRESHOLD = int(lines[9].split(':')[1])
		PERCENTAGE_NEXT_NEIGHBOR_FOR_MATCHES = float(lines[10].split(':')[1])
		OVERLAP_DISCARD_RATIO = float(lines[11].split(':')[1])
		TRANSFORMATION_SCALE_DISCARD_THRESHOLD = float(lines[12].split(':')[1])
		TRANSFORMATION_ANGLE_DISCARD_THRESHOLD = float(lines[13].split(':')[1])
		LETTUCE_AREA_THRESHOLD = int(lines[14].split(':')[1])
		CONTOUR_MATCHING_MIN_MATCH = int(lines[15].split(':')[1])
		ORTHO_SCALE = float(lines[16].split(':')[1])
		REDUCTION_FACTOR = ORTHO_SCALE/SCALE
		OPEN_MORPH_LID_SIZE = int(lines[17].split(':')[1])
		CLOSE_MORPH_LID_SIZE = int(lines[18].split(':')[1])
		

		FFT_PARALLEL_CORES_TO_USE = int(lines[19].split(':')[1])
		use_camera = lines[20].split(':')[1]
		override_sifts = (True if lines[21].split(':')[1] == 'true' or lines[21].split(':')[1] == 'True' else False)
		inside_radius_lettuce_matching_threshold = 200*SCALE
		number_of_rows_in_groups = int(lines[22].split(':')[1])
		groups_to_use = slice(0,None)
		patches_to_use = slice(0,None)
		is_single_group = (True if lines[23].split(':')[1] == 'true' or lines[23].split(':')[1] == 'True' else False)
		is_flir = True
		lid_detection_method = lines[26].split(':')[1]
		circle_error = int(lines[27].split(':')[1])
		lid_search_surrounding_patch_number = int(lines[28].split(':')[1])
		TRANSFORMATION_ERR_STD = float(lines[29].split(':')[1].split(',')[0]),float(lines[29].split(':')[1].split(',')[1])
		GPS_ERROR_STD = float(lines[30].split(':')[1].split(',')[0]),float(lines[30].split(':')[1].split(',')[1])
		LID_ERR_STD = float(lines[31].split(':')[1])

		lid_detection_model_path = lines[32].split(':')[1]

		save_coords_on_csv = (True if lines[33].split(':')[1] == 'true' or lines[33].split(':')[1] == 'True' else False)
		save_new_tiffs = (True if lines[34].split(':')[1] == 'true' or lines[34].split(':')[1] == 'True' else False)
		new_tiffs_path = '{0}/{1}/output_tiffs'.format(destination,scan_date)

		temp_lid_image_address = '{0}/lid_temp.png'.format(repository_address)

		patch_folder = '{0}'.format(bin2tif_address)
		SIFT_folder = '{0}/{1}/SIFT'.format(destination,scan_date)
		lid_file = '{0}'.format(lid_add)
		CORRECTED_coordinates_file = '{0}/{1}/{2}_coordinates_CORRECTED.csv'.format(destination,scan_date,scan_date)
		plot_npy_file = '{0}/{1}/plt.npy'.format(destination,scan_date)
		field_image_path = '{0}/{1}'.format(destination,scan_date)
		correction_log_file = '{0}/{1}/logs/log_{2}_at_{3}.csv'.format(destination,scan_date,method,datetime.datetime.now().strftime("%d-%m-%y_%H:%M"))
		scan_date_stng = scan_date
		lettuce_coords = None