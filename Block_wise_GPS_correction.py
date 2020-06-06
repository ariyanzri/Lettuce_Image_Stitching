import numpy as np
import cv2
import random
import math
import multiprocessing
import datetime
import sys
import gc
import pickle
import os
import threading
import socket
import statistics
import datetime

# from sklearn.linear_model import RANSACRegressor
# from sklearn.datasets import make_regression
# from sklearn.base import BaseEstimator
# from skimage.feature import hog

from Customized_myltiprocessing import MyPool
from heapq import heappush, heappop, heapify
from collections import OrderedDict,Counter


PATCH_SIZE = (3296, 2472)
PATCH_SIZE_GPS = (8.899999997424857e-06,1.0199999998405929e-05)
HEIGHT_RATIO_FOR_ROW_SEPARATION = 0.1
NUMBER_OF_ROWS_IN_GROUPS = 10
# NUMBER_OF_ROWS_IN_GROUPS = 4
PERCENTAGE_OF_GOOD_MATCHES_FOR_GROUP_WISE_CORRECTION = 0.5
GPS_TO_IMAGE_RATIO = (PATCH_SIZE_GPS[0]/PATCH_SIZE[1],PATCH_SIZE_GPS[1]/PATCH_SIZE[0])
MINIMUM_PERCENTAGE_OF_INLIERS = 0.1
MINIMUM_NUMBER_OF_MATCHES = 100
RANSAC_MAX_ITER = 1000
RANSAC_ERROR_THRESHOLD = 5
PERCENTAGE_NEXT_NEIGHBOR_FOR_MATCHES = 0.8
LETTUCE_AREA_THRESHOLD = 5000

GPS_ERROR_Y = 0.0000005
GPS_ERROR_X = 0.000001

FFT_PARALLEL_CORES_TO_USE = 20

def remove_shadow(image):

	hsvImg = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

	hsvImg[...,2] = np.amax(hsvImg[...,2])

	rgb_img = cv2.cvtColor(hsvImg,cv2.COLOR_HSV2BGR)

	# cv2.namedWindow('shd',cv2.WINDOW_NORMAL)
	# cv2.resizeWindow('shd', 500,500)
	# cv2.imshow('shd',rgb_img)
	# cv2.waitKey(0)
	
	return rgb_img

def adjust_gamma(image, gamma=1.0):

	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
	
	return cv2.LUT(image, table)

def convert_to_gray(img):
	
	# coefficients = [-1,1,2] 
	# m = np.array(coefficients).reshape((1,3))
	# img_g = cv2.transform(img, m)
		
	img_g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	# green_channel = img[:,:,1].copy()
	# red_channel = img[:,:,2].copy()
	# blue_channel = img[:,:,0].copy()

	# img = green_channel-0.61*blue_channel-0.39*red_channel

	# min_p = np.amin(img)
	# max_p = np.amax(img)
	# rng = (max_p-min_p)
	
	# img_g = cv2.normalize(img, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)

	return img_g

def load_preprocess_image(address):
	img = cv2.imread(address)
	
	img = img.astype('uint8')
	img_g = convert_to_gray(img)

	# cv2.namedWindow('fig1',cv2.WINDOW_NORMAL)
	# cv2.namedWindow('fig2',cv2.WINDOW_NORMAL)
	# cv2.resizeWindow('fig1', 500,500)
	# cv2.resizeWindow('fig2', 500,500)
	# cv2.imshow('fig1',img)
	# cv2.imshow('fig2',img_g)
	# cv2.waitKey(0)

	return img, img_g

def choose_SIFT_key_points(patch,x1,y1,x2,y2):
	global SIFT_folder

	kp = []
	desc = []

	kp_tmp = patch.SIFT_kp_locations
	desc_tmp = patch.SIFT_kp_desc

	for i,k in enumerate(kp_tmp):
		if k[0]>= x1 and k[0]<=x2 and k[1]>=y1 and k[1]<=y2:
			kp.append(k)

			desc.append(list(np.array(desc_tmp[i,:])))

	desc = np.array(desc)
	
	return kp,desc

def detect_SIFT_key_points(img,x1,y1,x2,y2):
	sift = cv2.xfeatures2d.SIFT_create()
	main_img = img.copy()
	img = img[y1:y2,x1:x2]
	kp,desc = sift.detectAndCompute(img,None)

	kp_n = []
	for k in kp:
		kp_n.append(cv2.KeyPoint(k.pt[0]+x1,k.pt[1]+y1,k.size))

	return kp_n,desc

def get_top_percentage_of_matches_no_KNN(p1,p2,desc1,desc2,kp1,kp2):

	bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
	# matches = bf.match(desc1,desc2)
	matches = bf.knnMatch(desc1,desc2, k=2)
	matches = sorted(matches, key = lambda x:x[0].distance)
	img3 = None
	img3 = cv2.drawMatches(p1.rgb_img,kp1,p2.rgb_img,kp2,[m[0] for m in matches[:20]],img3,matchColor=(0,255,0))
	cv2.namedWindow('fig',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('fig', 500,500)
	cv2.imshow('fig',img3)
	cv2.waitKey(0)

def get_good_matches(desc1,desc2):
	
	try:
		bf = cv2.BFMatcher()
	
		matches = bf.knnMatch(desc1,desc2, k=2)

		if len(matches)<=1:
			return None

		good = []
		for m in matches:
			if len(m)>=2 and m[0].distance <= PERCENTAGE_NEXT_NEIGHBOR_FOR_MATCHES*m[1].distance:
				good.append(m)
				
		matches = np.asarray(good)
		return matches
			
	except Exception as e:
		print('Error in get_good_matches: {0}'.format(e))
		return None

	

def get_translation_from_single_matches(x1,y1,x2,y2):
	x_t = x2-x1
	y_t = y2-y1

	return np.array([[1,0,x_t],[0,1,y_t],[0,0,1]])

# def calculate_error_for_translation(T,matches,kp1,kp2):
# 	error = 0

# 	for m in matches[:,0]:
# 		p1 = kp1[m.queryIdx]
# 		p2 = kp2[m.trainIdx]

# 		translated_p1 = T.dot([p1[0],p1[1],1]).astype(int)

# 		distance = math.sqrt((translated_p1[0]-p2[0])**2 + (translated_p1[1]-p2[1])**2)

# 		if distance>RANSAC_ERROR_THRESHOLD:
# 			error+=1

# 	return error

def calculate_error_for_translation(T,P1,P2):
	
	squared_diff = (T.dot(P1.T)-P2.T)**2
	squared_diff = squared_diff.T
	squared_diff = squared_diff[:,0:2]
	distances = np.sqrt(np.sum(squared_diff,axis=1))
	thresholded_distances = distances.copy()
	thresholded_distances[thresholded_distances<=RANSAC_ERROR_THRESHOLD] = 0
	thresholded_distances[thresholded_distances>RANSAC_ERROR_THRESHOLD] = 1

	return np.sum(thresholded_distances)

def ransac_parallel(i,matches,kp1,kp2,return_dict):
	m = matches[i,0]
	p1 = kp1[m.queryIdx]
	p2 = kp2[m.trainIdx]
	T = get_translation_from_single_matches(p1[0],p1[1],p2[0],p2[1])	
	error = calculate_error_for_translation(T,matches,kp1,kp2)

	return_dict[i] = (T,error)

def find_translation(matches,kp1,kp2):
	
	# max_possible_sampel = min(len(matches),RANSAC_MAX_ITER)

	# samples_indices = random.sample(range(0,len(matches)),max_possible_sampel)
	# manager = multiprocessing.Manager()
	# return_dict = manager.dict()
	# jobs = []

	# for i in samples_indices:
		
	# 	p = multiprocessing.Process(target=ransac_parallel, args=(i,matches,kp1,kp2,return_dict))
	# 	jobs.append(p)
	# 	p.daemon = False
	# 	p.start()		

	# for proc in jobs:
	# 	proc.join()

	# min_T = None
	# min_error = sys.maxsize
	# min_per_inlier = 100.0

	# for i in return_dict:
	# 	T,error = return_dict[i]

	# 	if error < min_error:
	# 		min_error = error
	# 		min_T = T
	# 		min_per_inlier = (len(matches)-error)/len(matches)

	# return min_T,min_per_inlier

	P1 = np.zeros((len(matches),3))
	P2 = np.zeros((len(matches),3))

	for i,m in enumerate(matches[:,0]):
		p1 = kp1[m.queryIdx]
		p2 = kp2[m.trainIdx]

		P1[i,0] = p1[0]
		P1[i,1] = p1[1]
		P1[i,2] = 1
		P2[i,0] = p2[0]
		P2[i,1] = p2[1]
		P2[i,2] = 1

	diff = P2-P1
	diff_x = diff[:,0]
	diff_y = diff[:,1]

	# max_possible_sampel = min(len(matches),RANSAC_MAX_ITER)
	# samples_indices = random.sample(range(0,len(matches)),max_possible_sampel)
	# min_T = None
	# min_error = sys.maxsize
	# min_per_inlier = 100.0

	# for i in samples_indices:
	# 	T = get_translation_from_single_matches(P1[i,0],P1[i,1],P2[i,0],P2[i,1])
	# 	error = calculate_error_for_translation(T,P1,P2)
	# 	if error < min_error:
	# 		min_error = error
	# 		min_T = T
	# 		min_per_inlier = (len(matches)-error)/len(matches)

	# return min_T,min_per_inlier

	import matplotlib.pyplot as plt

	plt.axis('equal')

	plt.scatter(diff_x,diff_y,alpha=0.2)

	plt.show()

	diff_x_counter = Counter(list(diff_x))
	t_x = diff_x_counter.most_common(1)[0][0]
	diff_y_counter = Counter(list(diff_y))
	t_y = diff_y_counter.most_common(1)[0][0]
	print(t_x)
	print(t_y)

	return np.array([[1,0,t_x],[0,1,t_y],[0,0,1]]),0

def find_scale_and_theta(H):

	a = H[0,0]
	b = H[1,0]

	s = math.sqrt(a**2+b**2)
	theta = math.degrees(math.acos(H[0,0]/s))

	return s,theta
	

def find_homography(matches,kp1,kp2,ov_2_on_1,ov_1_on_2):	
	
	if len(matches)>1:
		src = np.float32([ kp1[m.queryIdx] for m in matches[:,0] ]).reshape(-1,1,2)
		dst = np.float32([ kp2[m.trainIdx] for m in matches[:,0] ]).reshape(-1,1,2)
	else:
		return None,0,0,0

	H, masked = cv2.estimateAffinePartial2D(dst, src, maxIters = 1000, confidence = 0.99, refineIters = 5)

	if H is None or H.shape != (2,3):
		return None,0,0,0

	scale,theta = find_scale_and_theta(H)

	H = np.append(H,np.array([[0,0,1]]),axis=0)
	H[0:2,0:2] = np.array([[1,0],[0,1]])
	return H,np.sum(masked)/len(masked),scale,theta

def get_dissimilarity_on_overlaps(p1,p2,H):

	p1_ul = [0,0,1]
	p1_ur = [PATCH_SIZE[1],0,1]
	p1_ll = [0,PATCH_SIZE[0],1]
	p1_lr = [PATCH_SIZE[1],PATCH_SIZE[0],1]

	p1_ul_new = H.dot(p1_ul).astype(int)
	p1_ur_new = H.dot(p1_ur).astype(int)
	p1_ll_new = H.dot(p1_ll).astype(int)
	p1_lr_new = H.dot(p1_lr).astype(int)
	
	p1_x1 = 0
	p1_y1 = 0
	p1_x2 = PATCH_SIZE[1]
	p1_y2 = PATCH_SIZE[0]

	p2_x1 = 0
	p2_y1 = 0
	p2_x2 = PATCH_SIZE[1]
	p2_y2 = PATCH_SIZE[0]

	flag = False

	if p1_ul_new[0]>=0 and p1_ul_new[0]<PATCH_SIZE[1] and p1_ul_new[1]>=0 and p1_ul_new[1]<PATCH_SIZE[0]:
		p2_x1 = p1_ul_new[0]
		p2_y1 = p1_ul_new[1]

		p1_x2 = PATCH_SIZE[1] - p1_ul_new[0]
		p1_y2 = PATCH_SIZE[0] - p1_ul_new[1]

		flag = True

	if p1_ur_new[0]>=0 and p1_ur_new[0]<PATCH_SIZE[1] and p1_ur_new[1]>=0 and p1_ur_new[1]<PATCH_SIZE[0]:
		p2_x2 = p1_ur_new[0]
		p2_y1 = p1_ur_new[1]

		p1_x1 = PATCH_SIZE[1] - p1_ur_new[0]
		p1_y2 = PATCH_SIZE[0] - p1_ur_new[1]

		flag = True

	if p1_ll_new[0]>=0 and p1_ll_new[0]<PATCH_SIZE[1] and p1_ll_new[1]>=0 and p1_ll_new[1]<PATCH_SIZE[0]:
		p2_x1 = p1_ll_new[0]
		p2_y2 = p1_ll_new[1]

		p1_x2 = PATCH_SIZE[1] - p1_ll_new[0]
		p1_y1 = PATCH_SIZE[0] - p1_ll_new[1]

		flag = True

	if p1_lr_new[0]>=0 and p1_lr_new[0]<PATCH_SIZE[1] and p1_lr_new[1]>=0 and p1_lr_new[1]<PATCH_SIZE[0]:
		p2_x2 = p1_lr_new[0]
		p2_y2 = p1_lr_new[1]

		p1_x1 = PATCH_SIZE[1] - p1_lr_new[0]
		p1_y1 = PATCH_SIZE[0] - p1_lr_new[1]

		flag = True

	if not flag:
		return -1

	## XOR dissimilarity 

	p1.load_img()
	p2.load_img()
	overlap_1_img = p1.rgb_img[p1_y1:p1_y2,p1_x1:p1_x2,:]
	overlap_2_img = p2.rgb_img[p2_y1:p2_y2,p2_x1:p2_x2,:]

	shape_1 = np.shape(overlap_1_img)
	shape_2 = np.shape(overlap_2_img)

	if shape_1[0] == 0 or shape_1[1] == 0 or shape_2[0] == 0 or shape_2[1] == 0:
		return -1

	overlap_1_img = cv2.cvtColor(overlap_1_img, cv2.COLOR_BGR2GRAY)
	overlap_2_img = cv2.cvtColor(overlap_2_img, cv2.COLOR_BGR2GRAY)

	overlap_1_img = cv2.blur(overlap_1_img,(5,5))
	overlap_2_img = cv2.blur(overlap_2_img,(5,5))

	ret1,overlap_1_img = cv2.threshold(overlap_1_img,0,255,cv2.THRESH_OTSU)
	ret1,overlap_2_img = cv2.threshold(overlap_2_img,0,255,cv2.THRESH_OTSU)

	tmp_size = np.shape(overlap_1_img)
	
	overlap_1_img[overlap_1_img==255] = 1
	overlap_2_img[overlap_2_img==255] = 1

	xnor_images = np.logical_xor(overlap_1_img,overlap_2_img)

	dissimilarity = round(np.sum(xnor_images)/(tmp_size[0]*tmp_size[1]),2)
	
	p1.delete_img()
	p2.delete_img()

	## FFT dissimilarity 

	# fft1 = p1.get_fft_region(p1_x1,p1_y1,p1_x2,p1_y2)
	# fft2 = p2.get_fft_region(p2_x1,p2_y1,p2_x2,p2_y2)

	# dissimilarity = np.sqrt(np.sum((fft1-fft2)**2)/(fft1.shape[0]*fft1.shape[1]*fft1.shape[2]))

	# p1.delete_img()
	# p2.delete_img()

	## HOG dissimilarity

	# hog1 = p1.get_hog_region(p1_x1,p1_y1,p1_x2,p1_y2)
	# hog2 = p2.get_hog_region(p2_x1,p2_y1,p2_x2,p2_y2)

	# dissimilarity = np.sqrt(np.sum((hog1-hog2)**2)/(hog1.shape[0]))

	## RMSE simple

	# p1.load_img()
	# p2.load_img()

	# overlap_1_img = p1.rgb_img[p1_y1:p1_y2,p1_x1:p1_x2,:]
	# overlap_2_img = p2.rgb_img[p2_y1:p2_y2,p2_x1:p2_x2,:]

	# dissimilarity = np.sqrt(np.sum((overlap_1_img-overlap_2_img)**2)/(overlap_2_img.shape[0]*overlap_2_img.shape[1]*overlap_2_img.shape[2]))

	# p1.delete_img()
	# p2.delete_img()

	return dissimilarity

def visualize_plot():
	global plot_npy_file
	import matplotlib.pyplot as plt

	plt.axis('equal')

	data = np.load(plot_npy_file)

	c = []
	for d in data:
		c.append('red' if d[2] == 0 else 'green')

	plt.scatter(data[:,0],data[:,1],color=c,alpha=0.5)

	plt.show()

def report_time(start,end):
	print('-----------------------------------------------------------')
	print('Start date time: {0}\nEnd date time: {1}\nTotal running time: {2}.'.format(start,end,end-start))

def get_new_GPS_Coords(p1,p2,H):

	c1 = [0,0,1]
	
	c1 = H.dot(c1).astype(int)

	diff_x = -c1[0]
	diff_y = -c1[1]

	gps_scale_x = (PATCH_SIZE_GPS[0])/(PATCH_SIZE[1])
	gps_scale_y = -(PATCH_SIZE_GPS[1])/(PATCH_SIZE[0])

	diff_x = diff_x*gps_scale_x
	diff_y = diff_y*gps_scale_y

	new_UL = (round(p2.gps.UL_coord[0]-diff_x,7),round(p2.gps.UL_coord[1]-diff_y,7))

	diff_UL = (p1.gps.UL_coord[0]-new_UL[0],p1.gps.UL_coord[1]-new_UL[1])

	new_UR = (p1.gps.UR_coord[0]-diff_UL[0],p1.gps.UR_coord[1]-diff_UL[1])
	new_LL = (p1.gps.LL_coord[0]-diff_UL[0],p1.gps.LL_coord[1]-diff_UL[1])
	new_LR = (p1.gps.LR_coord[0]-diff_UL[0],p1.gps.LR_coord[1]-diff_UL[1])
	new_center = (p1.gps.Center[0]-diff_UL[0],p1.gps.Center[1]-diff_UL[1])

	new_coords = GPS_Coordinate(new_UL,new_UR,new_LL,new_LR,new_center)

	return new_coords

def get_new_GPS_Coords_for_groups(p1,p2,H):

	c1 = [0,0,1]
	
	c1 = H.dot(c1).astype(int)

	diff_x = -c1[0]
	diff_y = -c1[1]

	gps_scale_x = (PATCH_SIZE_GPS[0])/(PATCH_SIZE[1])
	gps_scale_y = -(PATCH_SIZE_GPS[1])/(PATCH_SIZE[0])

	diff_x = diff_x*gps_scale_x
	diff_y = diff_y*gps_scale_y

	# moved_UL = (round(p2.gps.UL_coord[0]-diff_x,7),round(p2.gps.UL_coord[1]-diff_y,7))

	# diff_UL = (p1.gps.UL_coord[0]-moved_UL[0],p1.gps.UL_coord[1]-moved_UL[1])

	diff_UL = (-diff_x,-diff_y)

	new_UL = (p1.gps.UL_coord[0]-diff_UL[0],p1.gps.UL_coord[1]-diff_UL[1])
	new_UR = (p1.gps.UR_coord[0]-diff_UL[0],p1.gps.UR_coord[1]-diff_UL[1])
	new_LL = (p1.gps.LL_coord[0]-diff_UL[0],p1.gps.LL_coord[1]-diff_UL[1])
	new_LR = (p1.gps.LR_coord[0]-diff_UL[0],p1.gps.LR_coord[1]-diff_UL[1])
	new_center = (p1.gps.Center[0]-diff_UL[0],p1.gps.Center[1]-diff_UL[1])

	new_coords = GPS_Coordinate(new_UL,new_UR,new_LL,new_LR,new_center)

	return new_coords

def correct_groups_internally_helper(args):

	return args[0].correct_internally(),args[0].group_id

# def correct_groups_internally_helper(gid,group,result_dict):

# 	result_dict[gid] = group.correct_internally()

def get_good_matches_based_on_GPS_error(desc1,desc2,kp1,kp2):
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(desc1,desc2, k=2)

	good = []
	for m in matches:
		p1 = kp1[m[0].queryIdx]
		p2 = kp2[m[0].trainIdx]

		if 	abs(p1[0]-p2[0])<=GPS_ERROR_X/GPS_TO_IMAGE_RATIO[0] and abs(p1[1]-p2[1])<=GPS_ERROR_Y/GPS_TO_IMAGE_RATIO[1]:
			good.append(m)

	sorted_matches = sorted(good, key=lambda x: x[0].distance)

	good = []

	# if len(sorted_matches)>NUMBER_OF_GOOD_MATCHES_FOR_GROUP_WISE_CORRECTION:
	# 	good += sorted_matches[0:NUMBER_OF_GOOD_MATCHES_FOR_GROUP_WISE_CORRECTION]
	# else:
	# 	good += sorted_matches

	number_of_good_matches = int(math.floor(len(sorted_matches)*PERCENTAGE_OF_GOOD_MATCHES_FOR_GROUP_WISE_CORRECTION))
	good = sorted_matches[0:number_of_good_matches]

	matches = np.asarray(good)

	return matches

def get_top_percentage_matches(desc1,desc2,kp1,kp2):
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(desc1,desc2, k=2)

	if matches is None or len(matches) == 0:
		return None

	if len(matches[0]) < 2:
		return None

	good = []
	for m in matches:

		if m[0].distance < PERCENTAGE_NEXT_NEIGHBOR_FOR_MATCHES*m[1].distance:
			good.append(m)

	sorted_matches = sorted(good, key=lambda x: x[0].distance)

	good = []

	# if len(sorted_matches)>NUMBER_OF_GOOD_MATCHES_FOR_GROUP_WISE_CORRECTION:
	# 	good += sorted_matches[0:NUMBER_OF_GOOD_MATCHES_FOR_GROUP_WISE_CORRECTION]
	# else:
	# 	good += sorted_matches

	number_of_good_matches = int(math.floor(len(sorted_matches)*PERCENTAGE_OF_GOOD_MATCHES_FOR_GROUP_WISE_CORRECTION))
	good = sorted_matches[0:number_of_good_matches]

	matches = np.asarray(good)

	return matches

def get_top_n_matches(desc1,desc2,kp1,kp2,n):
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(desc1,desc2, k=2)

	good = []
	for m in matches:
		
		if 	m[0].distance < PERCENTAGE_NEXT_NEIGHBOR_FOR_MATCHES*m[1].distance:
			good.append(m)

	sorted_matches = sorted(good, key=lambda x: x[0].distance)

	good = []

	# if len(sorted_matches)>NUMBER_OF_GOOD_MATCHES_FOR_GROUP_WISE_CORRECTION:
	# 	good += sorted_matches[0:NUMBER_OF_GOOD_MATCHES_FOR_GROUP_WISE_CORRECTION]
	# else:
	# 	good += sorted_matches

	number_of_good_matches = min(n,len(sorted_matches))
	good = sorted_matches[0:number_of_good_matches]

	matches = np.asarray(good)

	return matches

def calculate_homography_for_super_patches(kp,prev_kp,matches):

	src_list = []
	dst_list = []

	for i,mtch in enumerate(matches):
		src_list += [kp[i][m.queryIdx] for m in mtch[:,0]]
		dst_list += [prev_kp[i][m.trainIdx] for m in mtch[:,0]]

	src = np.float32(src_list).reshape(-1,1,2)
	dst = np.float32(dst_list).reshape(-1,1,2)
	
	H, masked = cv2.estimateAffinePartial2D(dst, src, maxIters = 9000, confidence = 0.999, refineIters = 15)
	
	H = np.append(H,np.array([[0,0,1]]),axis=0)
	H[0:2,0:2] = np.array([[1,0],[0,1]])
	
	return H

def get_corrected_string(patches):
	
	final_results = ''

	for p in patches:
		p.gps.UL_coord = (round(p.gps.UL_coord[0],7),round(p.gps.UL_coord[1],7))
		p.gps.LL_coord = (round(p.gps.LL_coord[0],7),round(p.gps.LL_coord[1],7))
		p.gps.UR_coord = (round(p.gps.UR_coord[0],7),round(p.gps.UR_coord[1],7))
		p.gps.LR_coord = (round(p.gps.LR_coord[0],7),round(p.gps.LR_coord[1],7))
		p.gps.Center = (round(p.gps.Center[0],7),round(p.gps.Center[1],7))

		final_results += '{:s},{:.7f},{:.7f},{:.7f},{:.7f},{:.7f},{:.7f},{:.7f},{:.7f},{:.7f},{:.7f}\n'\
		.format(p.name,p.gps.UL_coord[0],p.gps.UL_coord[1],p.gps.LL_coord[0],p.gps.LL_coord[1],p.gps.UR_coord[0],p.gps.UR_coord[1]\
			,p.gps.LR_coord[0],p.gps.LR_coord[1],p.gps.Center[0],p.gps.Center[1])

	return final_results


def get_result_dict_from_strings(s):

	res_dict = {}

	for l in s.split('\n'):
		if l == '':
			break
		
		features = l.split(',')

		filename = features[0]
		upper_left = (float(features[1]),float(features[2]))
		lower_left = (float(features[3]),float(features[4]))
		upper_right = (float(features[5]),float(features[6]))
		lower_right = (float(features[7]),float(features[8]))
		center = (float(features[9]),float(features[10]))

		coord = GPS_Coordinate(upper_left,upper_right,lower_left,lower_right,center)
		
		res_dict[filename] = coord
		

	return res_dict

# lid methods

def get_lids():
	global lid_file

	lids = {}

	with open(lid_file) as f:
		lines = f.read()

		for l in lines.split('\n'):
			if l == '':
				break

			features = l.split(',')

			marker = features[0]
			lat = features[1]
			lon = features[2]

			lids[marker] = (float(lon),float(lat))

	return lids

def get_name_of_patches_with_lids(lids,use_not_corrected=False):
	global CORRECTED_coordinates_file,coordinates_file

	patches_names_with_lid = []

	if use_not_corrected:
		address_of_coodinates = coordinates_file
	else:
		address_of_coodinates = CORRECTED_coordinates_file

	with open(address_of_coodinates) as f:
		lines = f.read()
		lines = lines.replace('"','')

		for l in lines.split('\n'):
			if l == '':
				break
			if l == 'Filename,Upper left,Lower left,Upper right,Lower right,Center' or l == 'name,upperleft,lowerleft,uperright,lowerright,center':
				continue

			features = l.split(',')

			filename = features[0]
			upper_left = (float(features[1]),float(features[2]))
			lower_left = (float(features[3]),float(features[4]))
			upper_right = (float(features[5]),float(features[6]))
			lower_right = (float(features[7]),float(features[8]))
			center = (float(features[9]),float(features[10]))

			coord = GPS_Coordinate(upper_left,upper_right,lower_left,lower_right,center)
			
			for l in lids:
				if coord.is_coord_inside(lids[l]) or coord.is_point_near(lids[l],PATCH_SIZE_GPS[1]):
					patches_names_with_lid.append((l,filename,coord))

	return patches_names_with_lid

def fit_circle(xs,ys):

	us = xs - np.mean(xs)
	vs = ys - np.mean(ys)

	A1 = np.sum(us**2)
	B1 = np.sum(us*vs)
	C1 = 0.5*(np.sum(us**3)+np.sum(us*(vs**2)))
	A2 = B1
	B2 = np.sum(vs**2)
	C2 = 0.5*(np.sum(vs**3)+np.sum(vs*(us**2)))

	v = (A1*C2 - A2*C1)/(A1*B2 - A2*B1)
	u = (C1-B1*v)/A1

	r = int(math.sqrt(u**2+v**2+(A1+B2)/np.shape(xs)[0]))

	x = int(u+np.mean(xs))
	y = int(v+np.mean(ys))

	return x,y,r

def circle_error(x,y,r,xs,ys):
	err = 0

	for i in range(0,np.shape(xs)[0]):
		d = math.sqrt((x-xs[i])**2+(y-ys[i])**2)
		if d>2*r:
			err+=0
		elif d<r/2:
			err+=0
		else:
			err += abs(d - r)

	return err

def ransac(xs,ys,iterations,number_of_points):
	best_x = -1
	best_y = -1
	best_r = -1
	min_error = None

	for i in range(0,iterations):
		
		indexes = random.sample(range(0,np.shape(xs)[0]),number_of_points)
	
		xs_r = xs[indexes]
		ys_r = ys[indexes]

		x,y,r = fit_circle(xs_r,ys_r)
		err = circle_error(x,y,r,xs,ys)

		if min_error == None or min_error>err:
			min_error = err
			best_x = x
			best_y = y
			best_r = r

	return best_x,best_y,best_r

def get_unique_lists(xs,ys):
	tmp, ind1 = np.unique(xs,return_index=True)
	tmp, ind2 = np.unique(ys,return_index=True)

	ind = np.intersect1d(ind1,ind2)

	return xs[ind],ys[ind]

def get_lid_in_patch(img_name,l,pname,coord,ransac_iter=100,ransac_min_num_fit=10):
	global patch_folder
	img = cv2.imread('{0}/{1}'.format(patch_folder,img_name))
	
	img[:,:,1:3] = 0

	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	img = adjust_gamma(img,2.5)
	
	max_intensity = np.amax(img)
	
	t = max_intensity-2
	
	(thresh, img) = cv2.threshold(img, t, 255, cv2.THRESH_BINARY)
	kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (200, 200))
	img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)	

	kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
	img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

	shp = np.shape(img)

	img, contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	
	new_contours = []

	for c in contours:
		for p in c:
			new_contours.append([p[0][0],p[0][1]])
	
	new_contours = np.array(new_contours)

	if np.shape(new_contours)[0]<ransac_min_num_fit:
		return -1,-1,-1,-1,-1,-1

	xs = np.array(new_contours[:,0])
	ys = np.array(new_contours[:,1])

	xs,ys = get_unique_lists(xs,ys)

	if np.shape(xs)[0]<ransac_min_num_fit:
		return -1,-1,-1,-1,-1,-1

	x,y,r = ransac(xs,ys,ransac_iter,ransac_min_num_fit)
	

	if r >= 400 and r <= 500:
		return x,y,r,l,pname,coord
	else:
		return -1,-1,-1,-1,-1,-1

	# if x >= 0 and x < shp[1] and y >= 0 and y < shp[0] and r >= 400 and r <= 500:
	# 	return x,y,r,l,pname,coord
	# else:
	# 	return -1,-1,-1,-1,-1,-1

def get_lid_in_patch_helper(args):
	return get_lid_in_patch(*args)

def calculate_error_of_correction(use_not_corrected=False):
	distances = []

	lids = get_lids()
	lid_patch_names = get_name_of_patches_with_lids(lids,use_not_corrected)

	args_list = []

	for l_marker,p_name,coord in lid_patch_names:
		args_list.append((p_name,l_marker,p_name,coord))

	processes = MyPool(no_of_cores_to_use)

	results = processes.map(get_lid_in_patch_helper,args_list)
	processes.close()

	for x,y,r,l,pn,crd in results:
		if r!=-1:
			old_lid = lids[l]

			patch = Patch(pn,crd)

			point = patch.convert_image_to_GPS_coordinate((x,y))
			print('--------')
			print(old_lid)
			print(point)
			print('--------')
			distances.append(math.sqrt((old_lid[0]-point[0])**2+(old_lid[1]-point[1])**2))
			
			
			# patch.load_img()
			# patch.visualize_with_single_GPS_point(point,(x+10,y+10),r)

	return statistics.mean(distances),statistics.stdev(distances)

# --------------- new method in which we consider all patches -------------------

def find_all_neighbors(patches,patch):

	neighbors = []

	for p in patches:
		if (p.has_overlap(patch) or patch.has_overlap(p)) and p != patch:
			overlap1 = patch.get_overlap_rectangle(p)
			overlap2 = p.get_overlap_rectangle(patch)
			
			if overlap1[2]-overlap1[0]<PATCH_SIZE[1]/5 and overlap1[3]-overlap1[1]<PATCH_SIZE[0]/5:
				continue

			neighbors.append(p)

	return neighbors

def draw_together(patches):
	
	up = patches[0].gps.UL_coord[1]
	down = patches[0].gps.LL_coord[1]
	left = patches[0].gps.UL_coord[0]
	right = patches[0].gps.UR_coord[0]

	for p in patches:
		if p.gps.UL_coord[1]>=up:
			up=p.gps.UL_coord[1]

		if p.gps.LL_coord[1]<=down:
			down=p.gps.LL_coord[1]

		if p.gps.UL_coord[0]<=left:
			left=p.gps.UL_coord[0]

		if p.gps.UR_coord[0]>=right:
			right=p.gps.UR_coord[0]


	super_patch_size = (int(math.ceil((up-down)/GPS_TO_IMAGE_RATIO[1]))+100,int(math.ceil((right-left)/GPS_TO_IMAGE_RATIO[0]))+100,3)
	UL = (left,up)

	result = np.zeros(super_patch_size)

	for p in patches:
		p.load_img()

		x_diff = p.gps.UL_coord[0] - UL[0]
		y_diff = UL[1] - p.gps.UL_coord[1]
		
		st_x = int(math.ceil(x_diff/GPS_TO_IMAGE_RATIO[0]))
		st_y = int(math.ceil(y_diff/GPS_TO_IMAGE_RATIO[1]))
		
		result[st_y:st_y+PATCH_SIZE[0],st_x:st_x+PATCH_SIZE[1],:] = p.rgb_img

		p.delete_img()

	result = np.array(result).astype('uint8')
	result = cv2.resize(result,(int(result.shape[1]/5),int(result.shape[0]/5)))

	cv2.imshow('fig',result)
	cv2.waitKey(0)

def merge_all_neighbors(corrected_neighbors,patch):
	total_kp = []
	total_desc = []

	up = corrected_neighbors[0].gps.UL_coord[1]
	down = corrected_neighbors[0].gps.LL_coord[1]
	left = corrected_neighbors[0].gps.UL_coord[0]
	right = corrected_neighbors[0].gps.UR_coord[0]

	for p in corrected_neighbors:
		if p.gps.UL_coord[1]>=up:
			up=p.gps.UL_coord[1]

		if p.gps.LL_coord[1]<=down:
			down=p.gps.LL_coord[1]

		if p.gps.UL_coord[0]<=left:
			left=p.gps.UL_coord[0]

		if p.gps.UR_coord[0]>=right:
			right=p.gps.UR_coord[0]


	super_patch_size = (int(math.ceil((up-down)/GPS_TO_IMAGE_RATIO[1]))+100,int(math.ceil((right-left)/GPS_TO_IMAGE_RATIO[0]))+100,3)
	UL = (left,up)

	result = np.zeros(super_patch_size)

	patch.load_SIFT_points()
	# patch.load_img()

	for p in corrected_neighbors:
		# p.load_img()
		p.load_SIFT_points()

		overlap = p.get_overlap_rectangle(patch)
		kp,desc = choose_SIFT_key_points(p,overlap[0],overlap[1],overlap[2],overlap[3])
		
		x_diff = p.gps.UL_coord[0] - UL[0]
		y_diff = UL[1] - p.gps.UL_coord[1]
		
		st_x = int(math.ceil(x_diff/GPS_TO_IMAGE_RATIO[0]))
		st_y = int(math.ceil(y_diff/GPS_TO_IMAGE_RATIO[1]))
		
		# result[st_y:st_y+PATCH_SIZE[0],st_x:st_x+PATCH_SIZE[1],:] = p.rgb_img
		for i,k in enumerate(kp):
			total_kp.append((k[0]+st_x,k[1]+st_y))
			total_desc.append(desc[i,:])
			# cv2.circle(result,(k[0]+st_x,k[1]+st_y),2,(0,0,255),-1)

		# p.delete_img()

	total_desc = np.array(total_desc)

	# result = np.array(result).astype('uint8')
	# result = cv2.resize(result,(int(result.shape[1]/5),int(result.shape[0]/5)))
	# img = patch.rgb_img.copy()
	# img = cv2.resize(img,(int(PATCH_SIZE[1]/5),int(PATCH_SIZE[0]/5)))
	# cv2.imshow('figmain',img)
	# cv2.imshow('fig',result)
	# cv2.waitKey(0)

	return UL,total_kp,total_desc

def get_new_GPS_Coords_all_neighbors(p1,UL,H):
	c1 = [0,0,1]
	
	c1 = H.dot(c1).astype(int)

	diff_x = -c1[0]
	diff_y = -c1[1]

	gps_scale_x = (PATCH_SIZE_GPS[0])/(PATCH_SIZE[1])
	gps_scale_y = -(PATCH_SIZE_GPS[1])/(PATCH_SIZE[0])

	diff_x = diff_x*gps_scale_x
	diff_y = diff_y*gps_scale_y

	new_UL = (UL[0]-diff_x,UL[1]-diff_y)

	diff_UL = (p1.gps.UL_coord[0]-new_UL[0],p1.gps.UL_coord[1]-new_UL[1])

	new_UR = (p1.gps.UR_coord[0]-diff_UL[0],p1.gps.UR_coord[1]-diff_UL[1])
	new_LL = (p1.gps.LL_coord[0]-diff_UL[0],p1.gps.LL_coord[1]-diff_UL[1])
	new_LR = (p1.gps.LR_coord[0]-diff_UL[0],p1.gps.LR_coord[1]-diff_UL[1])
	new_center = (p1.gps.Center[0]-diff_UL[0],p1.gps.Center[1]-diff_UL[1])

	new_coords = GPS_Coordinate(new_UL,new_UR,new_LL,new_LR,new_center)

	return new_coords

def add_to_gps_coord(gps,jx,jy):
	UL_coord = (gps.UL_coord[0]+jx,gps.UL_coord[1]+jy)
	UR_coord = (gps.UR_coord[0]+jx,gps.UR_coord[1]+jy)
	LL_coord = (gps.LL_coord[0]+jx,gps.LL_coord[1]+jy)
	LR_coord = (gps.LR_coord[0]+jx,gps.LR_coord[1]+jy)
	Center = (gps.Center[0]+jx,gps.Center[1]+jy)

	return GPS_Coordinate(UL_coord,UR_coord,LL_coord,LR_coord,Center)

def calculate_dissimilarity(p1,p2,p1_x1,p1_y1,p1_x2,p1_y2,p2_x1,p2_y1,p2_x2,p2_y2):
	overlap_1_img = p1.rgb_img[p1_y1:p1_y2,p1_x1:p1_x2,:]
	overlap_2_img = p2.rgb_img[p2_y1:p2_y2,p2_x1:p2_x2,:]

	shape_1 = np.shape(overlap_1_img)
	shape_2 = np.shape(overlap_2_img)

	if shape_1 != shape_2:
		# if shape_1[0]<shape_2[0]:
		# 	overlap_2_img = overlap_2_img[:shape_1[0],:,:]
		# 	shape_2 = shape_1
		# if shape_1[1]<shape_2[1]:
		# 	overlap_2_img = overlap_2_img[:,:shape_1[1],:]
		# 	shape_2 = shape_1
		
		# if shape_2[0]<shape_1[0]:
		# 	overlap_1_img = overlap_1_img[:shape_2[0],:,:]
		# 	shape_1 = shape_2
		# if shape_2[1]<shape_1[1]:
		# 	overlap_1_img = overlap_1_img[:,:shape_2[1],:]
		# 	shape_1 = shape_2

		if shape_1[0]*shape_1[1] > shape_2[0]*shape_2[1]:
			overlap_1_img = cv2.resize(overlap_1_img,(shape_2[1],shape_2[0]))
			shape_1 = shape_2
		else:
			overlap_2_img = cv2.resize(overlap_2_img,(shape_1[1],shape_1[0]))
			shape_2 = shape_1

	if shape_1[0] == 0 or shape_1[1] == 0 or shape_2[0] == 0 or shape_2[1] == 0:
		return sys.maxsize

	overlap_1_img = cv2.cvtColor(overlap_1_img, cv2.COLOR_BGR2GRAY)
	overlap_2_img = cv2.cvtColor(overlap_2_img, cv2.COLOR_BGR2GRAY)

	overlap_1_img = cv2.blur(overlap_1_img,(3,3))
	overlap_2_img = cv2.blur(overlap_2_img,(3,3))

	ret1,overlap_1_img = cv2.threshold(overlap_1_img,0,255,cv2.THRESH_OTSU)
	ret1,overlap_2_img = cv2.threshold(overlap_2_img,0,255,cv2.THRESH_OTSU)

	tmp_size = np.shape(overlap_1_img)
	
	overlap_1_img[overlap_1_img==255] = 1
	overlap_2_img[overlap_2_img==255] = 1

	xnor_images = np.logical_xor(overlap_1_img,overlap_2_img)

	dissimilarity = round(np.sum(xnor_images)/(tmp_size[0]*tmp_size[1]),2)
	# dissimilarity =  np.sum((overlap_1_img.astype("float") - overlap_2_img.astype("float")) ** 2)
	# dissimilarity /= float(overlap_1_img.shape[0] * overlap_1_img.shape[1])
	

	return dissimilarity

def calculate_average_dissimilarity(patch,neighbors):
	average_dissimilarity = 0

	patch.load_img()
	for n in neighbors:
		n.load_img()

		p1_x1,p1_y1,p1_x2,p1_y2 = patch.get_overlap_rectangle(n)
		p2_x1,p2_y1,p2_x2,p2_y2 = n.get_overlap_rectangle(patch)

		average_dissimilarity+= calculate_dissimilarity(patch,n,p1_x1,p1_y1,p1_x2,p1_y2,p2_x1,p2_y1,p2_x2,p2_y2)

		n.delete_img()

	patch.delete_img()

	average_dissimilarity/=len(neighbors)

	return average_dissimilarity

def jitter_and_calculate_dissimilarity(patch,neighbors,jx,jy):
	old_gps = patch.gps
	new_gps = add_to_gps_coord(patch.gps,jx,jy)
	patch.gps = new_gps

	average_dissimilarity = 0

	
	for n in neighbors:

		p1_x1,p1_y1,p1_x2,p1_y2 = patch.get_overlap_rectangle(n)
		p2_x1,p2_y1,p2_x2,p2_y2 = n.get_overlap_rectangle(patch)

		average_dissimilarity+= calculate_dissimilarity(patch,n,p1_x1,p1_y1,p1_x2,p1_y2,p2_x1,p2_y1,p2_x2,p2_y2)


	average_dissimilarity/=len(neighbors)

	patch.gps = old_gps

	return average_dissimilarity,new_gps

def jitter_image_to_find_least_dissimilarity(patch,neighbors):
	
	list_jitter_x = np.arange(-0.0000002, 0.0000002, 0.00000003)
	list_jitter_y = np.arange(-0.0000001, 0.0000001, 0.00000003)

	min_dissimilarity = sys.maxsize
	min_gps = None

	patch.load_img()
	for n in neighbors:
		n.load_img()

	for jx in list_jitter_x:
		for jy in list_jitter_y:
			dissimilarity,gps_jittered = jitter_and_calculate_dissimilarity(patch,neighbors,jx,jy)
			print(dissimilarity)

			if dissimilarity<min_dissimilarity:
				min_dissimilarity = dissimilarity
				min_gps = gps_jittered

	patch.delete_img()
	for n in neighbors:
		n.delete_img()

	return min_gps

def get_patch_with_max_number_of_corrected_neighbors(corrected,can_be_corrected_patches):

	neighbors = None
	best_patch = None
	score = 0

	can_be_corrected_patches = sorted(can_be_corrected_patches, key=lambda x: x.previously_checked)

	for patch in can_be_corrected_patches:

		corrected_neighbors = find_all_neighbors(corrected,patch)
		if len(corrected_neighbors)> score:
			score = len(corrected_neighbors)
			best_patch = patch
			neighbors = corrected_neighbors

	return best_patch,corrected_neighbors

def correct_patch_group_all_corrected_neighbors(group_id,patches):

	max_patch = patches[0]
	max_num = 0

	for p in patches:
		neighbors = find_all_neighbors(patches,p)
		if len(neighbors)>max_num:
			max_num = len(neighbors)
			max_patch = p

	# print(max_num)

	max_patch.Corrected = True
	max_patch.good_corrected = True
	can_be_corrected_patches = find_all_neighbors(patches,max_patch)

	while len(can_be_corrected_patches)>0:

		patch = can_be_corrected_patches.pop()

		tmp_neighbors = find_all_neighbors(patches,patch)

		# corrected_neighbors = find_all_neighbors(corrected_patches,patch)
		corrected_neighbors = [p for p in tmp_neighbors if p.Corrected]

		# patch, corrected_neighbors = get_patch_with_max_number_of_corrected_neighbors(corrected_patches,can_be_corrected_patches)
	
		# can_be_corrected_patches.remove(patch)

		if len(corrected_neighbors) == 0:
			# if not patch.previously_checked:
			# 	patch.previously_checked = True
			# 	print('Group {0} - Patch {1} NOT FIXED on {2} neighbors. Corrected Neighbors are empty.'.format(group_id,patch.name,len(corrected_neighbors)))
			# 	can_be_corrected_patches.insert(0,patch)
			# 	continue
			# else:
			# 	continue
			print('wierd')
			continue

		UL_merged, kp_merged, desc_merged = merge_all_neighbors(corrected_neighbors,patch)
		patch.load_SIFT_points()
		kp = patch.SIFT_kp_locations
		desc = patch.SIFT_kp_desc

		matches = get_good_matches(desc_merged,desc)

		H, perc_in,scale,theta = find_homography(matches,kp_merged,kp,None,None)

		if H is None:
			# if patch.previously_checked:
			# 	patch.Corrected = False
			# 	tmp_neighbors = find_all_neighbors(patches,patch)
			# 	can_be_corrected_patches+=[t for t in tmp_neighbors if t.Corrected == False and (t not in can_be_corrected_patches)]

			# 	print('Group {0} - Patch {1} fixed with H problem based on {2} neighbors.'.format(group_id,patch.name,len(corrected_neighbors)))
			# 	sys.stdout.flush()
			# 	continue
			# else:
			# 	patch.previously_checked = True
			# 	can_be_corrected_patches.insert(0,patch)
			# 	print('Group {0} - Patch {1} NOT FIXED {2} with neighbors. <Percentage Inliers:{3},# matches:{4}>. H IS NONE.'.format(group_id,patch.name,len(corrected_neighbors),perc_in,len(matches)))
			# 	continue
			continue

		coord = get_new_GPS_Coords_all_neighbors(patch,UL_merged,H)

		if (perc_in<MINIMUM_PERCENTAGE_OF_INLIERS or len(matches)<MINIMUM_NUMBER_OF_MATCHES):
			if patch.previously_checked == False:

				patch.previously_checked = True
				can_be_corrected_patches.insert(0,patch)
				print('Group {0} - Patch {1} NOT FIXED {2} with neighbors. <Percentage Inliers:{3},# matches:{4}>'.format(group_id,patch.name,len(corrected_neighbors),perc_in,len(matches)))
				continue  
			else:
				patch.gps = coord

				patch.good_corrected = False
				patch.Corrected = True

				print('Group {0} - Patch {1} fixed{2} based on {3} neighbors. <Percentage Inliers:{4},# matches:{5}>'.format(group_id,patch.name,'*' if patch.previously_checked else '',len(corrected_neighbors),perc_in,len(matches)))
				sys.stdout.flush()

		else:
			patch.gps = coord

			tmp_neighbors = find_all_neighbors(patches,patch)
			can_be_corrected_patches+=[t for t in tmp_neighbors if t.Corrected == False and (t not in can_be_corrected_patches)]

			patch.good_corrected = True
			patch.Corrected = True

			print('Group {0} - Patch {1} fixed{2} based on {3} neighbors. <Percentage Inliers:{4},# matches:{5}>'.format(group_id,patch.name,'*' if patch.previously_checked else '',len(corrected_neighbors),perc_in,len(matches)))
			sys.stdout.flush()


		

	return get_corrected_string(patches)


# ----------------------------------------------------------------------

def detect_SIFT_key_points(img,x1,y1,x2,y2):
	sift = cv2.xfeatures2d.SIFT_create()
	
	img = img[y1:y2,x1:x2]
	kp,desc = sift.detectAndCompute(img,None)

	kp_n = []
	for k in kp:
		kp_n.append(cv2.KeyPoint(k.pt[0]+x1,k.pt[1]+y1,k.size))

	kp = kp_n

	return kp_n,desc

def parallel_patch_creator(patch):
	
	global SIFT_folder,patch_folder

	if os.path.exists('{0}/{1}_SIFT.data'.format(SIFT_folder,patch.name.replace('.tif',''))):
		return

	patch.load_img()
	img = patch.rgb_img
	kp,desc = detect_SIFT_key_points(img,0,0,PATCH_SIZE[1],PATCH_SIZE[0])

	kp_tmp = [(p.pt[0], p.pt[1]) for p in kp]
	pickle.dump((kp_tmp,desc), open('{0}/{1}_SIFT.data'.format(SIFT_folder,patch.name.replace('.tif','')), "wb"))

	del kp,kp_tmp,desc
	patch.delete_img()

	print('Patch created and SIFT generated for {0}'.format(patch.name))
	sys.stdout.flush()
	

def parallel_patch_creator_helper(args):

	return parallel_patch_creator(*args)

def read_all_data():

	global patch_folder,coordinates_file
	patches = []

	with open(coordinates_file) as f:
		lines = f.read()
		lines = lines.replace('"','')

		for l in lines.split('\n'):
			if l == '':
				break
			if l == 'Filename,Upper left,Lower left,Upper right,Lower right,Center' or l == 'name,upperleft,lowerleft,uperright,lowerright,center':
				continue

			features = l.split(',')

			rgb,img = load_preprocess_image('{0}/{1}'.format(patch_folder,features[0]))

			upper_left = (float(features[1]),float(features[2]))
			lower_left = (float(features[3]),float(features[4]))
			upper_right = (float(features[5]),float(features[6]))
			lower_right = (float(features[7]),float(features[8]))
			center = (float(features[9]),float(features[10]))

			coord = GPS_Coordinate(upper_left,upper_right,lower_left,lower_right,center)

			patch = Patch(features[0],coord)
			patches.append(patch)

	return patches

def get_pairwise_transformation_info_helper(p,n,return_dict):

	return_dict[n.name] = (p.get_pairwise_transformation_info(n),n)

def jitter_and_calculate_fft(p1,neighbors,jx,jy):
	old_gps = p1.gps
	p1.gps = add_to_gps_coord(p1.gps,jx,jy)

	sum_differences = 0

	for p2 in neighbors:

		overlap_1,overlap_2 = p1.get_overlap_rectangles(p2)

		if overlap_1[2]-overlap_1[0] == 0 or overlap_1[3]-overlap_1[1] == 0 or \
		overlap_2[2]-overlap_2[0] == 0 or overlap_2[3]-overlap_2[1] == 0:
			continue

		fft1 = p1.get_fft_region(overlap_1[0],overlap_1[1],overlap_1[2],overlap_1[3])
		fft2 = p2.get_fft_region(overlap_2[0],overlap_2[1],overlap_2[2],overlap_2[3])

		if fft1 is None or fft2 is None:
			continue

		fft_difference = np.sqrt(np.sum((fft1-fft2)**2)/(fft1.shape[0]*fft1.shape[1]*fft1.shape[2]))
		sum_differences+=fft_difference

	# print(sum_differences)

	new_gps = p1.gps
	p1.gps = old_gps

	return sum_differences,new_gps

def jitter_and_calculate_fft_helper(args):
	return jitter_and_calculate_fft(*args)

def read_lettuce_heads_coordinates():
	global lettuce_heads_coordinates_file
	from numpy import genfromtxt

	lettuce_coords = genfromtxt(lettuce_heads_coordinates_file, delimiter=',',skip_header=1)

	col1 = lettuce_coords[:,0].copy()
	lettuce_coords[:,0] = lettuce_coords[:,1].copy()
	lettuce_coords[:,1] = col1

	return lettuce_coords

def calculate_average_min_distance_lettuce_heads(contour_centers,inside_lettuce_heads,T):
	
	average_distance = 0

	for c in contour_centers:
		min_distance = sys.maxsize

		for l in inside_lettuce_heads:
			new_l = (l[0]-T[0,2],l[1]-T[1,2])
			
			distance = math.sqrt((c[0]-new_l[0])**2+(c[1]-new_l[1])**2)

			if distance<min_distance:
				min_distance = distance

		average_distance+=min_distance

	average_distance/=len(contour_centers)

	return average_distance

def calculate_remaining_contour_matches_error(matches,T):
	average_distance = 0

	for m in matches:
		c1 = m[0]
		c2 = m[1]

		new_c2 = (c2[0]-T[0,2],c2[1]-T[1,2])
			
		distance = math.sqrt((c1[0]-new_c2[0])**2+(c1[1]-new_c2[1])**2)

		average_distance+=distance

	average_distance/=len(matches)
			

	return average_distance

def get_gps_diff_from_H(p1,p2,H):
	c1 = [0,0,1]
	
	c1 = H.dot(c1).astype(int)

	diff_x = -c1[0]
	diff_y = -c1[1]

	gps_scale_x = (PATCH_SIZE_GPS[0])/(PATCH_SIZE[1])
	gps_scale_y = -(PATCH_SIZE_GPS[1])/(PATCH_SIZE[0])

	diff_x = diff_x*gps_scale_x
	diff_y = diff_y*gps_scale_y

	new_UL = (round(p2.gps.UL_coord[0]-diff_x,7),round(p2.gps.UL_coord[1]-diff_y,7))

	diff_UL = (p1.gps.UL_coord[0]-new_UL[0],p1.gps.UL_coord[1]-new_UL[1])

	return diff_UL


def super_patch_pool_merging_method(patches,gid):
	super_patches = []
	for p in patches:
		super_patches.append(Super_Patch([p]))

	i = 1

	while len(super_patches)>1:
		new_supper_patches = []

		while len(super_patches)>0:

			sp1 = super_patches.pop()

			sp2,params,scr = sp1.find_best_super_patch_for_merging(super_patches)

			if sp2 is None:
				new_supper_patches.append(sp1)
				continue

			print('Group {0}: Merge accepted using score {1}.'.format(gid,scr))

			super_patches.remove(sp2)

			diff = sp1.get_total_gps_diff_from_params(sp2,params)
			sp2.correct_based_on_best_diff(diff)

			new_sp = Super_Patch(sp1.patches+sp2.patches)
			new_supper_patches.append(new_sp)

		print('Group {0}: Super Patches with group size {1} merged together.'.format(gid,i))
		sys.stdout.flush()

		i*=2

		super_patches = new_supper_patches.copy()

	return super_patches[0].patches


def GPS_distance(point1,point2):
	return math.sqrt((point2[0]-point1[0])**2+(point2[1]-point1[1])**2)


def get_best_neighbor_hybrid_method(p1,corrected):

	corrected_neighbors = [p for p in corrected if (p.has_overlap(p1) or p1.has_overlap(p))]

	best_score = sys.maxsize
	best_params = None
	best_p = None

	p1.load_SIFT_points()

	for p_tmp in corrected_neighbors:
		p_tmp.load_SIFT_points()

		params = p_tmp.get_pairwise_transformation_info(p1)
		
		if params is not None and params.dissimilarity < best_score:
			best_score = params.dissimilarity
			best_params = params
			best_p = p_tmp

	return best_p,best_params


def hybrid_method_UAV_lettuce_matching_step(patches,gid):
	global lettuce_coords

	not_corrected = []
	corrected = []
	step = 0

	for p in patches:
		
		old_gps = p.gps

		err = p.correct_based_on_contours_and_lettuce_heads(lettuce_coords)

		if err >=500:
			not_corrected.append(p)
		else:
			print('Group ID {0}: patch {1} corrected with {2} error.'.format(gid,p.name,err))
			sys.stdout.flush()
			corrected.append(p)

			gps_diff = (old_gps.UL_coord[0]-p.gps.UL_coord[0],old_gps.UL_coord[1]-p.gps.UL_coord[1])
			params = Neighbor_Parameters(None,None,None,None,None,None,None,None)
			logger(p,gps_diff,params,gid,step)
			step+=1

	return corrected,not_corrected,step

def hybrid_method_sift_correction_step(corrected,not_corrected,gid,starting_step):
	
	print('Group ID {0}: ---- Entering SIFT Correction Phase ----'.format(gid))
	sys.stdout.flush()

	step = starting_step

	while len(not_corrected)>0:
		
		p1 = not_corrected.pop()
		
		p2,params = get_best_neighbor_hybrid_method(p1,corrected)

		if p2 is None:
			print('Group ID {0}: ERROR- patch {1} has no good corrected neighbor and will be pushed back.'.format(gid,p1.name))
			sys.stdout.flush()
			not_corrected.insert(0,p1)
			continue

		H = params.H

		new_gps = get_new_GPS_Coords(p1,p2,H)

		gps_diff = (p1.gps.UL_coord[0]-new_gps.UL_coord[0],p1.gps.UL_coord[1]-new_gps.UL_coord[1])
		
		p1.gps = new_gps

		corrected.append(p1)

		logger(p1,gps_diff,params,gid,step)

		step+=1

		print('Group ID {0}: patch {1} corrected with {2} dissimilarity.'.format(gid,p1.name,params.dissimilarity))
		sys.stdout.flush()

	return corrected



class GPS_Coordinate:
	
	def __init__(self,UL_coord,UR_coord,LL_coord,LR_coord,Center):
		self.UL_coord = UL_coord
		self.UR_coord = UR_coord
		self.LL_coord = LL_coord
		self.LR_coord = LR_coord
		self.Center = Center

	def is_coord_inside(self, coord):
		if coord[0]>=self.UL_coord[0] and coord[0]<=self.UR_coord[0] and coord[1]<=self.UL_coord[1] and coord[1]>=self.LL_coord[1]:
			return True
		else:
			return False

	def is_point_near(self,point,threshold):
		if GPS_distance(self.UL_coord,point)<threshold or GPS_distance(self.UL_coord,point)<threshold or \
		GPS_distance(self.UL_coord,point)<threshold or	GPS_distance(self.UL_coord,point)<threshold:
			return True
		else:
			return False

	def is_coord_in_GPS_error_proximity(self,coord):
		if coord[0]>=self.UL_coord[0] and coord[0]<=self.UR_coord[0] and (abs(coord[1]-self.LL_coord[1])<GPS_ERROR_Y*2 or abs(coord[1]-self.UL_coord[1])<GPS_ERROR_Y*2):
			return True

		if coord[1]<=self.UL_coord[1] and coord[1]>=self.LL_coord[1] and (abs(coord[0]-self.LL_coord[0])<GPS_ERROR_X*2 or abs(coord[0]-self.LR_coord[0])<GPS_ERROR_X*2):
			return True

		return False

	def to_csv(self):
		return '{0};{1};{2};{3};{4};{5};{6};{7};{8};{9}'.format(self.UL_coord[0],self.UL_coord[1],self.UR_coord[0],self.UR_coord[1],\
			self.LL_coord[0],self.LL_coord[1],self.LR_coord[0],self.LR_coord[1],self.Center[0],self.Center[1])

class Graph():

	def __init__(self,no_vertex,vertex_names,gid=-1):
		self.vertecis_number = no_vertex
		self.vertex_index_to_name_dict = {}
		self.vertex_name_to_index_dict = {}
		for i,v in enumerate(vertex_names):
			self.vertex_index_to_name_dict[i] = v
			self.vertex_name_to_index_dict[v] = i

		self.edges = [[-1 for column in range(no_vertex)] for row in range(no_vertex)]
		self.gid = gid

	def initialize_edge_weights(self,patches):
		
		for p in patches:
			for n in p.neighbors:
				
				if self.edges[self.vertex_name_to_index_dict[p.name]][self.vertex_name_to_index_dict[n[0].name]] == -1:
					self.edges[self.vertex_name_to_index_dict[p.name]][self.vertex_name_to_index_dict[n[0].name]] = round(n[1].dissimilarity,2)
					self.edges[self.vertex_name_to_index_dict[n[0].name]][self.vertex_name_to_index_dict[p.name]] =  round(n[1].dissimilarity,2)
				else:
					if n[1].dissimilarity > self.edges[self.vertex_name_to_index_dict[p.name]][self.vertex_name_to_index_dict[n[0].name]]:
						self.edges[self.vertex_name_to_index_dict[p.name]][self.vertex_name_to_index_dict[n[0].name]] = round(n[1].dissimilarity,2)
						self.edges[self.vertex_name_to_index_dict[n[0].name]][self.vertex_name_to_index_dict[p.name]] = round(n[1].dissimilarity,2)


	def find_min_key(self,keys,mstSet):
		min_value = sys.maxsize

		for v in range(self.vertecis_number): 
			if keys[v] < min_value and mstSet[v] == False: 
				min_value = keys[v] 
				min_index = v 

		return min_index 


	def generate_MST_prim(self,starting_vertex):
		keys = [sys.maxsize]*self.vertecis_number
		parents = [None]*self.vertecis_number
		mstSet = [False]*self.vertecis_number

		keys[self.vertex_name_to_index_dict[starting_vertex]] = 0
		parents[self.vertex_name_to_index_dict[starting_vertex]] = -1

		for count in range(self.vertecis_number):
			u = self.find_min_key(keys,mstSet)
			mstSet[u] = True

			for v in range(self.vertecis_number):
				if self.edges[u][v] != -1 and mstSet[v] == False and keys[v] > self.edges[u][v]:
					keys[v] = self.edges[u][v]
					parents[v] = u

		
		return parents

	def get_patches_dict(self,patches):
		dict_patches={}

		for p in patches:
			dict_patches[p.name] = p

		return dict_patches

	def revise_GPS_from_generated_MST(self,patches,parents):
		dict_patches = self.get_patches_dict(patches)

		queue_traverse = []
		
		for v,p in enumerate(parents):
			if p == -1:
				queue_traverse = [v]
				break
		
		step = 0

		while len(queue_traverse) > 0:
			u = queue_traverse.pop()

			for v,p in enumerate(parents):
				if p == u:
					queue_traverse = [v] + queue_traverse

					patch = dict_patches[self.vertex_index_to_name_dict[v]]
					parent_patch = dict_patches[self.vertex_index_to_name_dict[p]]
					# H = [n[1].H for n in parent_patch.neighbors if n[0] == patch]
					# H = H[0]
					params = [n[1] for n in parent_patch.neighbors if n[0] == patch]
					param = params[0]
					H = param.H

					new_gps = get_new_GPS_Coords(patch,parent_patch,H)

					gps_diff = (patch.gps.UL_coord[0]-new_gps.UL_coord[0],patch.gps.UL_coord[1]-new_gps.UL_coord[1])
					# print(gps_diff)
					
					patch.gps = new_gps

					logger(patch,gps_diff,param,self.gid,step)
					
					step+=1

		string_corrected = get_corrected_string(patches)
		return string_corrected

class Neighbor_Parameters:
	def __init__(self,o_p,o_n,h,nm,pi,d,scale,theta):

		self.overlap_on_patch = o_p
		self.overlap_on_neighbor = o_n
		self.H = h
		self.num_matches = nm
		self.percentage_inliers = pi
		self.dissimilarity = d
		self.degrees = theta
		self.scale = scale


class Patch:
	
	def __init__(self,name,coord):
		self.name = name
		self.gps = coord
		self.neighbors = []
		self.SIFT_kp_locations = []
		self.SIFT_kp_desc = []
		self.previously_checked = False
		self.Corrected = False
		self.good_corrected = False
		self.rgb_img = None
		self.gray_img = None


	def __eq__(self,other):

		return (self.name == other.name)

	def __str__(self):
		return self.name

	def has_overlap(self,p):
		if self.gps.is_coord_inside(p.gps.UL_coord) or self.gps.is_coord_inside(p.gps.UR_coord) or\
			self.gps.is_coord_inside(p.gps.LL_coord) or self.gps.is_coord_inside(p.gps.LR_coord):
			return True
		else:
			return False

	def load_SIFT_points(self):
		global SIFT_folder

		if len(self.SIFT_kp_locations) == 0:
			(kp_tmp,desc_tmp) = pickle.load(open('{0}/{1}_SIFT.data'.format(SIFT_folder,self.name.replace('.tif','')), "rb"))
			self.SIFT_kp_locations = kp_tmp.copy()
			self.SIFT_kp_desc = desc_tmp.copy()

	def delete_SIFT_points(self):
		self.SIFT_kp_locations = None
		self.SIFT_kp_desc = None

		gc.collect()


	def load_img(self):
		global patch_folder

		img,img_g = load_preprocess_image('{0}/{1}'.format(patch_folder,self.name))
		self.rgb_img = img
		self.gray_img = img_g

	def delete_img(self):

		self.rgb_img = None
		self.gray_img = None

		gc.collect()

	def convert_image_to_GPS_coordinate(self,point):
		x_ratio = point[1]/PATCH_SIZE[1]
		y_ratio = point[0]/PATCH_SIZE[0]

		return (self.gps.UL_coord[0]+x_ratio,self.gps.UL_coord[1]-y_ratio)

	def get_hog_region(self,x1,y1,x2,y2):
		
		if self.rgb_img is None:
			self.load_img()

		img = self.rgb_img[y1:y2,x1:x2]

		fd = hog(img, orientations=8, pixels_per_cell=(8, 8),cells_per_block=(1, 1), visualize=False, multichannel=True)
		# cv2.imshow('fig 1',img)
		# cv2.imshow('fig 2',hog_image)
		# cv2.waitKey(0)

		self.delete_img()

		return np.array(fd)

	def get_overlap_rectangle(self,patch,increase_size=True):
		p1_x = 0
		p1_y = 0
		p2_x = PATCH_SIZE[1]
		p2_y = PATCH_SIZE[0]

		detect_overlap = False

		if patch.gps.UL_coord[1]>=self.gps.LL_coord[1] and patch.gps.UL_coord[1]<=self.gps.UL_coord[1]:
			detect_overlap = True
			p1_y = int(math.ceil(((patch.gps.UL_coord[1]-self.gps.UL_coord[1]) / (self.gps.LL_coord[1]-self.gps.UL_coord[1])*PATCH_SIZE[0])))
		
		if patch.gps.LL_coord[1]>=self.gps.LL_coord[1] and patch.gps.LL_coord[1]<=self.gps.UL_coord[1]:
			detect_overlap = True
			p2_y = int(math.ceil(((patch.gps.LR_coord[1]-self.gps.UL_coord[1]) / (self.gps.LL_coord[1]-self.gps.UL_coord[1])*PATCH_SIZE[0])))

		if patch.gps.UR_coord[0]<=self.gps.UR_coord[0] and patch.gps.UR_coord[0]>=self.gps.UL_coord[0]:
			detect_overlap = True
			p2_x = int(math.ceil(((patch.gps.UR_coord[0]-self.gps.UL_coord[0]) / (self.gps.UR_coord[0]-self.gps.UL_coord[0])*PATCH_SIZE[1])))
			
		if patch.gps.UL_coord[0]<=self.gps.UR_coord[0] and patch.gps.UL_coord[0]>=self.gps.UL_coord[0]:
			detect_overlap = True
			p1_x = int(math.ceil(((patch.gps.LL_coord[0]-self.gps.UL_coord[0]) / (self.gps.UR_coord[0]-self.gps.UL_coord[0])*PATCH_SIZE[1])))
			
		if patch.gps.is_coord_inside(self.gps.UL_coord) and patch.gps.is_coord_inside(self.gps.UR_coord) and \
		patch.gps.is_coord_inside(self.gps.LL_coord) and patch.gps.is_coord_inside(self.gps.LR_coord):
			p1_x = 0
			p1_y = 0
			p2_x = PATCH_SIZE[1]
			p2_y = PATCH_SIZE[0]
			detect_overlap = True

		if increase_size:
			if p1_x>0+PATCH_SIZE[1]/10:
				p1_x-=PATCH_SIZE[1]/10

			if p2_x<9*PATCH_SIZE[1]/10:
				p2_x+=PATCH_SIZE[1]/10

			if p1_y>0+PATCH_SIZE[0]/10:
				p1_y-=PATCH_SIZE[0]/10

			if p2_y<9*PATCH_SIZE[0]/10:
				p2_y+=PATCH_SIZE[0]/10

		if detect_overlap == False:
			return 0,0,0,0

		return int(p1_x),int(p1_y),int(p2_x),int(p2_y)

	def get_overlap_rectangles(self,patch,increase_size=True):
		
		p1_x1 = 0
		p1_y1 = 0
		p1_x2 = PATCH_SIZE[1]
		p1_y2 = PATCH_SIZE[0]

		p2_x1 = 0
		p2_y1 = 0
		p2_x2 = PATCH_SIZE[1]
		p2_y2 = PATCH_SIZE[0]

		if patch.gps.UL_coord[1]>=self.gps.LL_coord[1] and patch.gps.UL_coord[1]<=self.gps.UL_coord[1]:
			p1_y1 = int(math.ceil((self.gps.UL_coord[1]-patch.gps.UL_coord[1])/GPS_TO_IMAGE_RATIO[1]))
			p2_y2 = PATCH_SIZE[0]-p1_y1
		
		if patch.gps.LL_coord[1]>=self.gps.LL_coord[1] and patch.gps.LL_coord[1]<=self.gps.UL_coord[1]:
			p1_y2 = int(math.ceil((self.gps.UL_coord[1]-patch.gps.LL_coord[1])/GPS_TO_IMAGE_RATIO[1]))
			p2_y1 = PATCH_SIZE[0]-p1_y2

		if patch.gps.UR_coord[0]<=self.gps.UR_coord[0] and patch.gps.UR_coord[0]>=self.gps.UL_coord[0]:
			p1_x2 = int(math.ceil((patch.gps.UR_coord[0]-self.gps.UL_coord[0])/GPS_TO_IMAGE_RATIO[0]))
			p2_x1 = PATCH_SIZE[1]-p1_x2

		if patch.gps.UL_coord[0]<=self.gps.UR_coord[0] and patch.gps.UL_coord[0]>=self.gps.UL_coord[0]:
			p1_x1 = int(math.ceil((patch.gps.UL_coord[0]-self.gps.UL_coord[0])/GPS_TO_IMAGE_RATIO[0]))
			p2_x2 = PATCH_SIZE[1]-p1_x1

		return (p1_x1,p1_y1,p1_x2,p1_y2),(p2_x1,p2_y1,p2_x2,p2_y2)


	def visualize_with_single_GPS_point(self,point,point_img,r):
		if self.rgb_img is None:
			return

		output = self.rgb_img.copy()
		cv2.circle(output,point_img,20,(0,255,0),thickness=-1)
		cv2.circle(output,point_img,r,(255,0,0),thickness=15)


		ratio_x = (point[0] - self.gps.UL_coord[0])/(self.gps.UR_coord[0]-self.gps.UL_coord[0])
		ratio_y = (self.gps.UL_coord[1] - point[1])/(self.gps.UL_coord[1]-self.gps.LL_coord[1])

		shp = np.shape(output)
		cv2.circle(output,(int(ratio_x*shp[1]),int(ratio_y*shp[0])),20,(0,0,255),thickness=-1)

		cv2.namedWindow('GPS',cv2.WINDOW_NORMAL)
		cv2.resizeWindow('GPS', 500,500)
		cv2.imshow('GPS',output)
		cv2.waitKey(0)

	def get_pairwise_transformation_info(self,neighbor):
		overlap1,overlap2 = neighbor.get_overlap_rectangles(self)
		
		if overlap1[2]-overlap1[0]<PATCH_SIZE[1]/5 and overlap1[3]-overlap1[1]<PATCH_SIZE[0]/5:
			
			return None

		kp1,desc1 = choose_SIFT_key_points(neighbor,overlap1[0],overlap1[1],overlap1[2],overlap1[3])
		kp2,desc2 = choose_SIFT_key_points(self,overlap2[0],overlap2[1],overlap2[2],overlap2[3])

		if desc1 is None or len(desc1) == 0 or desc2 is None or len(desc2) == 0:
			return None

		# matches = get_good_matches(desc2,desc1)
		matches = get_top_percentage_matches(desc2,desc1,kp2,kp1)
		# matches = get_top_n_matches(desc2,desc1,kp2,kp1,50)
		# matches = get_good_matches_based_on_GPS_error(desc2,desc1,kp2,kp1)

		if matches is None or len(matches) == 0:
			# print('match is none or len matches is 0.')
			return None

		num_matches = len(matches)

		H,percentage_inliers,scale,theta = find_homography(matches,kp2,kp1,overlap1,overlap2)

		# if percentage_inliers<0.10 or num_matches<100:
		# 	return None

		# H,percentage_inliers = find_translation(matches,kp2,kp1)

		# print(percentage_inliers)

		if H is None:
			# print('H is none.')
			return None

		percentage_inliers = round(percentage_inliers*100,2)

		dissimilarity = get_dissimilarity_on_overlaps(neighbor,self,H)
		# dissimilarity = - percentage_inliers*num_matches

		if dissimilarity == -1:
			
			return None
		
		# print(percentage_inliers,num_matches,dissimilarity,(overlap1[2]-overlap1[0])*(overlap1[3]-overlap1[1]))

		return Neighbor_Parameters(overlap2,overlap1,H,num_matches,percentage_inliers,dissimilarity,scale,theta)

	def get_fft_region(self,x1,y1,x2,y2):
		
		if self.rgb_img is None:
			self.load_img()

		img = self.rgb_img[y1:y2,x1:x2,:]

		if img is None or len(img.shape) !=3 or img.shape[0] == 0 or img.shape[1] == 0 or img.shape[2] == 0:
			return None

		f = np.fft.fft2(img)
		fshift = np.fft.fftshift(f)
		
		zeros = fshift==0
		fshift[zeros] = 1e-10
		
		magnitude_spectrum = 20*np.log(np.abs(fshift))
		
		magnitude_spectrum[zeros] = 0

		# magnitude_spectrum = f

		return magnitude_spectrum.astype('uint8')

	def correct_based_on_neighbors(self,neighbors):

		list_jitter_x = np.arange(-GPS_ERROR_X, GPS_ERROR_X, 0.0000001)
		list_jitter_y = np.arange(-GPS_ERROR_Y, GPS_ERROR_Y, 0.0000001)

		self.load_img()
		for n in neighbors:
			n.load_img()

		old_gps = self.gps

		args_list = []

		for jx in list_jitter_x:
			for jy in list_jitter_y:
				
				args_list.append((self,neighbors,jx,jy))

		process = MyPool(FFT_PARALLEL_CORES_TO_USE)
		result = process.map(jitter_and_calculate_fft_helper,args_list)

		self.delete_img()
		for n in neighbors:
			n.delete_img()
		
		min_dissimilarity = sys.maxsize
		min_gps = None

		for fft_difference,gps_current in result:
			if fft_difference == 0:
				continue

			if fft_difference<min_dissimilarity:
				min_dissimilarity = fft_difference
				min_gps = gps_current

		return min_gps

	def get_all_contours(self,overlap=None):
		if self.rgb_img is None:
			self.load_img()
		
		img = self.gray_img.copy()
		
		if overlap is not None:
			img = img[overlap[1]:overlap[3],overlap[0]:overlap[2]]


		ret1,img = cv2.threshold(img,0,255,cv2.THRESH_OTSU)

		kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
		img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

		kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
		img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)		

		cv2.namedWindow('figg',cv2.WINDOW_NORMAL)
		cv2.resizeWindow('figg', 500,500)
		cv2.imshow('figg',img)
		cv2.waitKey(0)

		image, contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		
		# contours_new = []
		# for cnt in contours:
		# 	contours_new.append(cnt+(overlap[0],overlap[1]))

		# cv2.drawContours(self.rgb_img, contours, -1, (0,255,0),10)

		return contours

	def get_lettuce_contours(self,list_lettuce_heads=None,overlap=None):
		if self.rgb_img is None:
			self.load_img()

		img = remove_shadow(self.rgb_img.copy())
		
		if overlap is not None:
			img = img[overlap[1]:overlap[3],overlap[0]:overlap[2]]

		green_channel = img[:,:,1].copy()
		red_channel = img[:,:,2].copy()
		blue_channel = img[:,:,0].copy()

		img = green_channel-0.61*blue_channel-0.39*red_channel

		min_p = np.amin(img)
		max_p = np.amax(img)
		rng = (max_p-min_p)
		

		img = cv2.normalize(img, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
		
		img[img>=150] = 255
		img[img<150] = 0

		img  = cv2.medianBlur(img,17)

		kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50,50))
		img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)	

		kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50,50))
		img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)		

		

		image, contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		
		final_contours = []

		for cnt in contours:
			area = cv2.contourArea(cnt)
			if area>=LETTUCE_AREA_THRESHOLD:
				final_contours.append(cnt)


		# contours_new = []
		# for cnt in contours:
		# 	contours_new.append(cnt+(overlap[0],overlap[1]))

		# cv2.drawContours(self.rgb_img, final_contours, -1, (0,255,0),10)

		# cv2.namedWindow('gr',cv2.WINDOW_NORMAL)
		# cv2.resizeWindow('gr', 500,500)
		# cv2.imshow('gr',self.rgb_img)
		# cv2.waitKey(0)

		return final_contours

	def get_lettuce_contours_centers(self,list_lettuce_heads=None):
		
		contours = self.get_lettuce_contours(list_lettuce_heads)

		contour_centers = []

		for c in contours:
			M = cv2.moments(c)
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
			contour_centers.append((cX,cY))
			# cv2.circle(self.rgb_img, (cX, cY), 20, (0, 255, 0), -1)

		# for coord in list_lettuce_heads:
		# 	if self.gps.is_coord_inside(coord):

		# 		pX = int(abs(coord[0]-self.gps.UL_coord[0])/GPS_TO_IMAGE_RATIO[0])
		# 		pY = int(abs(coord[1]-self.gps.UL_coord[1])/GPS_TO_IMAGE_RATIO[1])
		# 		cv2.circle(self.rgb_img, (pX, pY), 20, (0, 0, 255 ), -1)
			
		# cv2.namedWindow('fig',cv2.WINDOW_NORMAL)
		# cv2.namedWindow('gr',cv2.WINDOW_NORMAL)
		# cv2.resizeWindow('fig', 500,500)
		# cv2.resizeWindow('gr', 500,500)

		# cv2.imshow('fig',self.rgb_img)
		# cv2.imshow('gr',img)
		# cv2.waitKey(0)

		return contour_centers

	def correct_based_on_matched_contour_centers(self,p2):

		self.load_img()
		p2.load_img()
		overlap_1,overlap_2 = self.get_overlap_rectangles(p2)


		contours1 = self.get_lettuce_contours(overlap=overlap_1)
		contours2 = p2.get_lettuce_contours(overlap=overlap_2)
		# contours1 = self.get_all_contours(overlap=overlap_1)
		# contours2 = p2.get_all_contours(overlap=overlap_2)

		pairs = []

		for i,cnt1 in enumerate(contours1):
			for j,cnt2 in enumerate(contours2):
				scr = cv2.matchShapes(cnt1,cnt2,1,0.0)
				
				pairs.append((i,j,scr))
		
		sorted_pairs = sorted(pairs, key = lambda x:x[2])
		used_i = []
		used_j = []
		matches = []

		for p in sorted_pairs:
			if p[0] in used_i or p[1] in used_j:
				continue

			M = cv2.moments(contours1[p[0]])
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
			center_1 = (cX,cY)

			M = cv2.moments(contours2[p[1]])
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
			center_2 = (cX,cY)


			matches.append((center_1,center_2))
			used_i.append(p[0])
			used_j.append(p[1])

			r = random.randint(0,256)
			g = random.randint(0,256)
			b = random.randint(0,256)

			cnt1 = contours1[p[0]]+(overlap_1[0],overlap_1[1])
			cnt2 = contours2[p[1]]+(overlap_2[0],overlap_2[1])

			cv2.drawContours(self.rgb_img, cnt1, -1, (b,g,r),10)
			cv2.drawContours(p2.rgb_img, cnt2, -1, (b,g,r),10)

		best_T = None
		best_error = sys.maxsize

		for m in matches:
			c1 = m[0]
			c2 = m[1]
			T = get_translation_from_single_matches(c1[0],c1[1],c2[0],c2[1])
			
			if abs(T[0,2])>=GPS_ERROR_X/GPS_TO_IMAGE_RATIO[0] or abs(T[1,2])>=GPS_ERROR_Y/GPS_TO_IMAGE_RATIO[1]:
					continue

			mean_error = calculate_remaining_contour_matches_error(matches,T)

			if mean_error<best_error:
				best_error = mean_error
				best_T = T
				

		if best_T is not None:
			self.move_GPS_based_on_lettuce(best_T)

		cv2.namedWindow('img1',cv2.WINDOW_NORMAL)
		cv2.namedWindow('img2',cv2.WINDOW_NORMAL)
		cv2.resizeWindow('img1', 500,500)
		cv2.resizeWindow('img2', 500,500)
		cv2.imshow('img1',self.rgb_img)
		cv2.imshow('img2',p2.rgb_img)
		cv2.waitKey(0)

		self.delete_img()
		p2.delete_img()
		
	def correct_based_on_contours_and_lettuce_heads(self,list_lettuce_heads):
		self.load_img()

		# cv2.namedWindow('fig',cv2.WINDOW_NORMAL)
		# cv2.resizeWindow('fig', 500,500)

		# cv2.imshow('fig',self.rgb_img)
		# cv2.waitKey(0)
		
		contour_centers = self.get_lettuce_contours_centers()
		inside_lettuce_heads = []

		for coord in list_lettuce_heads:
			if self.gps.is_coord_inside(coord) or self.gps.is_coord_in_GPS_error_proximity(coord):

				pX = int(abs(coord[0]-self.gps.UL_coord[0])/GPS_TO_IMAGE_RATIO[0])
				pY = int(abs(coord[1]-self.gps.UL_coord[1])/GPS_TO_IMAGE_RATIO[1])
				inside_lettuce_heads.append((pX,pY))

		# cv2.namedWindow('reg',cv2.WINDOW_NORMAL)
		# cv2.resizeWindow('reg', 500,500)

		# imgg = self.rgb_img.copy()

		# for c in contour_centers:
		# 	cv2.circle(imgg, (c[0], c[1]), 20, (0, 255, 0), -1)

		# for l in inside_lettuce_heads:
		# 	cv2.circle(imgg, (l[0], l[1]), 20, (0, 0, 255 ), -1)
			

		# cv2.imshow('reg',imgg)
		# cv2.waitKey(0)


		best_T = None
		best_error = sys.maxsize

		for c in contour_centers:
			for l in inside_lettuce_heads:

				T = get_translation_from_single_matches(c[0],c[1],l[0],l[1])

				if abs(T[0,2])>=GPS_ERROR_X/GPS_TO_IMAGE_RATIO[0] or abs(T[1,2])>=GPS_ERROR_Y/GPS_TO_IMAGE_RATIO[1]:
					continue

				mean_error = calculate_average_min_distance_lettuce_heads(contour_centers,inside_lettuce_heads,T)

				if mean_error<best_error:
					best_error = mean_error
					best_T = T

		if best_T is not None:
			self.move_GPS_based_on_lettuce(best_T)

		# imgg = self.rgb_img.copy()

		# for c in contour_centers:
		# 	cv2.circle(imgg, (c[0], c[1]), 20, (0, 255, 0), -1)

		# inside_lettuce_heads = []

		# for coord in list_lettuce_heads:
		# 	if self.gps.is_coord_inside(coord):

		# 		pX = int(abs(coord[0]-self.gps.UL_coord[0])/GPS_TO_IMAGE_RATIO[0])
		# 		pY = int(abs(coord[1]-self.gps.UL_coord[1])/GPS_TO_IMAGE_RATIO[1])
		# 		inside_lettuce_heads.append((pX,pY))

		# for l in inside_lettuce_heads:
		# 	cv2.circle(imgg, (l[0], l[1]), 20, (0, 0, 255 ), -1)
			
		# cv2.imshow('reg',imgg)
		# cv2.waitKey(0)
		self.delete_img()
		return best_error

	def move_GPS_based_on_lettuce(self,T):
		diff_x = -T[0,2]*GPS_TO_IMAGE_RATIO[0]
		diff_y = T[1,2]*GPS_TO_IMAGE_RATIO[1]
		diff = (diff_x,diff_y)

		new_UL = (self.gps.UL_coord[0]-diff[0],self.gps.UL_coord[1]-diff[1])
		new_UR = (self.gps.UR_coord[0]-diff[0],self.gps.UR_coord[1]-diff[1])
		new_LL = (self.gps.LL_coord[0]-diff[0],self.gps.LL_coord[1]-diff[1])
		new_LR = (self.gps.LR_coord[0]-diff[0],self.gps.LR_coord[1]-diff[1])
		new_center = (self.gps.Center[0]-diff[0],self.gps.Center[1]-diff[1])

		self.gps = GPS_Coordinate(new_UL,new_UR,new_LL,new_LR,new_center)


class Super_Patch:
	def __init__(self,patches):
		self.patches = patches

	def has_overlap(self,sp):
		for p1 in self.patches:
			for p2 in sp.patches:
				if p1.has_overlap(p2) or p2.has_overlap(p1):
					return True

		return False

	def number_of_patch_overlaps(self,sp):
		n = 0

		for p1 in self.patches:
			for p2 in sp.patches:
				if p1.has_overlap(p2) or p2.has_overlap(p1):
					n+=1

		return n

	def calculate_merge_score(self,sp):
		
		number_overlaped_patches = self.number_of_patch_overlaps(sp)
		# print(number_overlaped_patches)

		if number_overlaped_patches == 1:
			total_number_inliers = 0
			list_parameters = {}

			for p1 in self.patches:
				for p2 in sp.patches:
					if p1.has_overlap(p2) and p2.has_overlap(p1):
						tr_parameter = p1.get_pairwise_transformation_info(p2)
						if tr_parameter is None:
							list_parameters['{0}{1}'.format(p1.name,p2.name)] = tr_parameter
							continue
							
						number_inliers = tr_parameter.percentage_inliers * tr_parameter.num_matches
						total_number_inliers += number_inliers
						list_parameters['{0}{1}'.format(p1.name,p2.name)] = tr_parameter

			return total_number_inliers,list_parameters
		else:
			
			list_diff_x = []
			list_diff_y = []
			list_parameters = {}

			for p1 in self.patches:
				for p2 in sp.patches:
					if p1.has_overlap(p2) or p2.has_overlap(p1):
						tr_parameter = p1.get_pairwise_transformation_info(p2)
						if tr_parameter is None:
							list_parameters['{0}{1}'.format(p1.name,p2.name)] = tr_parameter
							continue

						gps_diff = get_gps_diff_from_H(p2,p1,tr_parameter.H)
						list_diff_x.append(gps_diff[0])
						list_diff_y.append(gps_diff[1])

						list_parameters['{0}{1}'.format(p1.name,p2.name)] = tr_parameter

			if len(list_diff_x)<2:
				return -sys.maxsize,list_parameters

			avg_stdev = (statistics.stdev(list_diff_x)+statistics.stdev(list_diff_y))/2

			return -avg_stdev,list_parameters


	def find_best_super_patch_for_merging(self,super_patches):
		best_sp = None
		best_score = 0
		best_params = None

		for sp in super_patches:
			if self.has_overlap(sp):
				score,params = self.calculate_merge_score(sp)
				
				if 0 > score:
					if best_score < 0:
						
						if score < best_score:
							best_score = score
							best_sp = sp
							best_params = params

					else:
						best_score = score
						best_sp = sp
						best_params = params

				elif score>0:

					if 0 > best_score >= -5e-7:
						continue
					else:

						if score>best_score:
							best_score = score
							best_sp = sp
							best_params = params

		return best_sp,best_params,best_score


	def get_total_gps_diff_from_params(self,best_sp,params):
		gps_diff_list = []

		for p1 in self.patches:
			for p2 in best_sp.patches:
				if not p1.has_overlap(p2) or not p2.has_overlap(p1):
					continue

				param_current = params['{0}{1}'.format(p1.name,p2.name)]
				if param_current is None:
					continue

				gps_diff = get_gps_diff_from_H(p2,p1,param_current.H)
				gps_diff_list.append(gps_diff)

		best_score = sys.maxsize
		best_diff = None

		for gps_diff in gps_diff_list:
			average_absolute_diff_sumed = 0

			for rem_gps_diff in gps_diff_list:
				average_absolute_diff_sumed+=abs(rem_gps_diff[0]-gps_diff[0])+abs(rem_gps_diff[1]-gps_diff[1])

			average_absolute_diff_sumed/=(2*len(gps_diff_list))

			if average_absolute_diff_sumed<best_score:
				best_score = average_absolute_diff_sumed
				best_diff = gps_diff

		return best_diff

	def correct_based_on_best_diff(self,diff):
		diff_x = diff[0]
		diff_y = diff[1]

		for p in self.patches:

			new_UL = (p.gps.UL_coord[0]-diff_x,p.gps.UL_coord[1]-diff_y)
			new_UR = (p.gps.UR_coord[0]-diff_x,p.gps.UR_coord[1]-diff_y)
			new_LL = (p.gps.LL_coord[0]-diff_x,p.gps.LL_coord[1]-diff_y)
			new_LR = (p.gps.LR_coord[0]-diff_x,p.gps.LR_coord[1]-diff_y)
			new_center = (p.gps.Center[0]-diff_x,p.gps.Center[1]-diff_y)

			new_coords = GPS_Coordinate(new_UL,new_UR,new_LL,new_LR,new_center)

			p.gps = new_coords


class Row:
	def __init__(self,patches):
		sorted_patches = sorted(patches, key=lambda x: x.gps.Center[0])
		self.patches = sorted_patches

	def correct_row_by_matching_lettuce_contours(self):
		previous_patch = None

		for i,patch in enumerate(self.patches):
			if i == 0:
				previous_patch = patch
				continue

			patch.correct_based_on_matched_contour_centers(previous_patch)

			previous_patch = patch
				

class Group:
	def __init__(self,gid,rows):
		self.group_id = gid
		self.patches = []

		new_rows = []
		for row in rows:
			new_row = [Patch(p.name,p.gps) for p in row]
			new_rows.append(new_row)
			self.patches += new_row

		self.rows = new_rows
		

	def load_all_patches_SIFT_points(self):
		for p in self.patches:
			p.load_SIFT_points()

		print('SIFT for all patches in group {0} loaded.'.format(self.group_id))
		sys.stdout.flush()

	def delete_all_patches_SIFT_points(self):
		for p in self.patches:
			p.delete_SIFT_points()

		print('SIFT for all patches in group {0} deleted.'.format(self.group_id))
		sys.stdout.flush()

	def pre_calculate_internal_neighbors_and_transformation_parameters(self,print_flg=True):
		remove_neighbors = []

		for p in self.patches:

			for n in self.patches:

				if n != p and (p.has_overlap(n) or n.has_overlap(p)):

					neighbor_param = p.get_pairwise_transformation_info(n)
					
					if neighbor_param is None:
						remove_neighbors.append((n,p))
						continue
					
					p.neighbors.append((n,neighbor_param))

			if print_flg:
				print('GROPU ID: {0} - Calculated Transformation and error values for {1} neighbors of {2}'.format(self.group_id,len(p.neighbors),p.name))

			sys.stdout.flush()

		for a,b in remove_neighbors:
			new_neighbors = []

			for n in a.neighbors:
				if b != n[0]:
					new_neighbors.append(n)
				
			a.neighbors = new_neighbors

		# remove_neighbors = []

		# for p in self.patches:

		# 	manager = multiprocessing.Manager()
		# 	return_dict = manager.dict()
		# 	jobs = []

		# 	for n in self.patches:

		# 		if n != p and (p.has_overlap(n) or n.has_overlap(p)):

		# 			process = multiprocessing.Process(target=get_pairwise_transformation_info_helper, args=(p,n,return_dict))
		# 			jobs.append(process)
		# 			process.daemon = False
		# 			process.start()	
					
		# 	for proc in jobs:
		# 		proc.join()

		# 	for name in return_dict:
		# 		neighbor_param = return_dict[name][0]
		# 		n = return_dict[name][1]
		# 		if neighbor_param == None:
		# 			remove_neighbors.append((n,p))
		# 			continue

		# 		p.neighbors.append((n,neighbor_param))

		# 	print('GROPU ID: {0} - Calculated Transformation and error values for {1} neighbors of {2}'.format(self.group_id,len(p.neighbors),p.name))
		# 	sys.stdout.flush()

		# for a,b in remove_neighbors:
		# 	new_neighbors = []

		# 	for n in a.neighbors:
		# 		if b != n[0]:
		# 			new_neighbors.append(n)
				
		# 	a.neighbors = new_neighbors


	def correct_row_by_row(self):

		# self.load_all_patches_SIFT_points()

		for i,r in enumerate(self.rows):

			for j,patch in enumerate(r):
				# print(i,j)
				if i == 0 and j == 0:
					patch.Corrected = True
					continue
				elif i == 0 and j>0:
					left_side_neighbor = r[j-1]
					down_side_neighbors = []
					neighbors = down_side_neighbors+[left_side_neighbor]
				elif i>0 and j==0:
					left_side_neighbor = None
					down_side_neighbors = find_all_neighbors(self.rows[i-1],patch)
					neighbors = down_side_neighbors
				elif i>0 and j>0:
					left_side_neighbor = r[j-1]
					down_side_neighbors = find_all_neighbors(self.rows[i-1],patch)
					neighbors = down_side_neighbors

				patch.gps = patch.correct_based_on_neighbors(neighbors)

				# patch.load_img()
				# main = cv2.resize(patch.rgb_img,(int(PATCH_SIZE[1]/5),int(PATCH_SIZE[0]/5)))
				# cv2.imshow('main',main)

				# draw_together(neighbors+[patch])

				# UL_merged, kp_merged, desc_merged = merge_all_neighbors(neighbors,patch)
				
				# kp = patch.SIFT_kp_locations
				# desc = patch.SIFT_kp_desc

				# matches = get_top_percentage_matches(desc_merged,desc,kp_merged,kp)

				# H, perc_in,scale,theta = find_homography(matches,kp_merged,kp,None,None)

				# if H is not None:

				# 	coord = get_new_GPS_Coords_all_neighbors(patch,UL_merged,H)

				# 	patch.gps = coord
				
				# 	patch.Corrected = True

				# 	print('Group {0} - Patch {1} fixed based on {2} neighbors. <Percentage Inliers:{3},# matches:{4}>'.format(self.group_id,patch.name,len(neighbors),perc_in,len(matches)))
				# else:
				# 	print('Group {0} - Patch {1} NOT FIXED on {2} neighbors. H is None. <Percentage Inliers:{3},# matches:{4}>'.format(self.group_id,patch.name,len(neighbors),perc_in,len(matches)))

				# draw_together(neighbors+[patch])
			
			print('Group ID {0}: row {1} corrected.'.format(self.group_id,i+1))
			sys.stdout.flush()

		# self.delete_all_patches_SIFT_points()

		return get_corrected_string(self.patches)

	def correct_internally(self):

		global lettuce_coords,no_of_cores_to_use

		print('Group {0} with {1} rows and {2} patches internally correction started.'.format(self.group_id,len(self.rows),len(self.patches)))
		
		# MST method

		# self.load_all_patches_SIFT_points()

		# self.pre_calculate_internal_neighbors_and_transformation_parameters()

		# G = Graph(len(self.patches),[p.name for p in self.patches],self.group_id)
		# G.initialize_edge_weights(self.patches)

		# try:
		# 	parents = G.generate_MST_prim(self.rows[0][0].name)
		# 	string_res = G.revise_GPS_from_generated_MST(self.patches,parents)
		# except Exception as e:
		# 	print(e)
		# 	string_res = get_corrected_string(self.patches)

		# self.delete_all_patches_SIFT_points()


		# string_res = self.correct_row_by_row()
		# string_res = correct_patch_group_all_corrected_neighbors(self.group_id,self.patches)

		# lettuce head matching (UAV)

		# for p in self.patches:
		# 	err = p.correct_based_on_contours_and_lettuce_heads(lettuce_coords)
		# 	print('Group ID {0}: patch {1} corrected with {2} error.'.format(self.group_id,p.name,err))
		# 	sys.stdout.flush()
		
		# string_res = get_corrected_string(self.patches)

		# Hybrid method: Lettuce head matching (UAV) and SIFT on remaining

		corrected,not_corrected,step = hybrid_method_UAV_lettuce_matching_step(self.patches,self.group_id)
			
		final_patches = hybrid_method_sift_correction_step(corrected,not_corrected,self.group_id,step)

		string_res = get_corrected_string(final_patches)

		# self.load_all_patches_SIFT_points()

		# corrected_patches = super_patch_pool_merging_method(self.patches,self.group_id)

		# string_res = get_corrected_string(self.patches)
		# self.delete_all_patches_SIFT_points()


		print('Group {0} was corrected internally. '.format(self.group_id))
		sys.stdout.flush()

		return string_res


	def correct_self_based_on_previous_group(self,previous_group):

		diff_x = []
		diff_y = []

		for i,patch_self in enumerate(self.rows[0]):

			patch_prev = previous_group.rows[NUMBER_OF_ROWS_IN_GROUPS-1][i]

			diff = (patch_self.gps.UL_coord[0] - patch_prev.gps.UL_coord[0],patch_self.gps.UL_coord[1] - patch_prev.gps.UL_coord[1])
			
			diff_x.append(diff[0])
			diff_y.append(diff[1])
		
		diff = (max(set(diff_x), key=diff_x.count),max(set(diff_y), key=diff_y.count))

		for p in self.patches:

			new_UL = (p.gps.UL_coord[0]-diff[0],p.gps.UL_coord[1]-diff[1])
			new_UR = (p.gps.UR_coord[0]-diff[0],p.gps.UR_coord[1]-diff[1])
			new_LL = (p.gps.LL_coord[0]-diff[0],p.gps.LL_coord[1]-diff[1])
			new_LR = (p.gps.LR_coord[0]-diff[0],p.gps.LR_coord[1]-diff[1])
			new_center = (p.gps.Center[0]-diff[0],p.gps.Center[1]-diff[1])

			new_coords = GPS_Coordinate(new_UL,new_UR,new_LL,new_LR,new_center)

			p.gps = new_coords

		print('Block {0} corrected based on previous block.'.format(self.group_id))
		sys.stdout.flush()

		# matches = []
		# kp = []
		# desc = []
		# prev_kp = []
		# prev_desc = []

		# for self_patch in self.rows[0]:
			
		# 	for prev_patch in previous_group.rows[-1]:

		# 		if self_patch.has_overlap(prev_patch) or prev_patch.has_overlap(self_patch):
		# 			overlap1 = self_patch.get_overlap_rectangle(prev_patch)
		# 			overlap2 = prev_patch.get_overlap_rectangle(self_patch)

		# 			kp1,desc1 = choose_SIFT_key_points(self_patch,overlap1[0],overlap1[1],overlap1[2],overlap1[3])
		# 			kp2,desc2 = choose_SIFT_key_points(prev_patch,overlap2[0],overlap2[1],overlap2[2],overlap2[3])

		# 			# print('overlap detected. {0}-\n\t{1}'.format(overlap1,overlap2))

		# 			kp.append(kp1)
		# 			desc.append(desc1)
		# 			prev_kp.append(kp2)
		# 			prev_desc.append(desc2)
					
		# 			matches.append(get_top_percentage_matches(desc2,desc1,kp2,kp1))

		# H = calculate_homography_for_super_patches(prev_kp,kp,matches)

		# base_patch_from_prev = previous_group.rows[-1][0]

		# for patch in self.patches:

		# 	patch.gps = get_new_GPS_Coords_for_groups(patch,base_patch_from_prev,H)

		# print('Block {0} corrected based on previous block.'.format(self.group_id))
		# sys.stdout.flush()

class Field:
	def __init__(self):
		global coordinates_file

		self.groups = self.initialize_field()
		print([g.group_id for g in self.groups])
		# for group in self.groups:
		# 	group.load_all_patches_SIFT_points()
		
	def initialize_field(self):
		global coordinates_file

		rows = self.get_rows()

		groups = []

		start = 0
		end = NUMBER_OF_ROWS_IN_GROUPS

		while start<len(rows):
			
			if end >= len(rows):
				end = len(rows)
				row_window = rows[start:end]
				group = Group(len(groups),row_window)
				groups.append(group)
				break
					
			row_window = rows[start:end]

			group = Group(len(groups),row_window)
			groups.append(group)

			start = end-1
			end = start + NUMBER_OF_ROWS_IN_GROUPS

		print('Field initialized with {0} groups of {1} rows each.'.format(len(groups),NUMBER_OF_ROWS_IN_GROUPS))
		sys.stdout.flush()

		return groups

	def get_rows(self):
		global coordinates_file

		center_of_rows = []
		patches = []
		

		with open(coordinates_file) as f:
			lines = f.read()
			lines = lines.replace('"','')

			for l in lines.split('\n'):
				if l == '':
					break
				if l == 'Filename,Upper left,Lower left,Upper right,Lower right,Center' or l == 'name,upperleft,lowerleft,uperright,lowerright,center':
					continue

				features = l.split(',')

				filename = features[0]
				upper_left = (float(features[1]),float(features[2]))
				lower_left = (float(features[3]),float(features[4]))
				upper_right = (float(features[5]),float(features[6]))
				lower_right = (float(features[7]),float(features[8]))
				center = (float(features[9]),float(features[10]))
				
				coord = GPS_Coordinate(upper_left,upper_right,lower_left,lower_right,center)
				patches.append(Patch(filename,coord))

				is_new = True

				for c in center_of_rows:
					if abs(center[1]-c[1]) < PATCH_SIZE_GPS[1]*HEIGHT_RATIO_FOR_ROW_SEPARATION:
						is_new = False

				if is_new:
					center_of_rows.append(center)

		patches_groups_by_rows = OrderedDict({})

		center_of_rows = sorted(center_of_rows, key=lambda x: x[1])

		for c in center_of_rows:
			patches_groups_by_rows[c] = []

		for p in patches:
			min_distance = PATCH_SIZE_GPS[1]*2
			min_row = None

			for c in center_of_rows:
				distance = abs(p.gps.Center[1]-c[1])
				if distance<min_distance:
					min_distance = distance
					min_row = c

			patches_groups_by_rows[min_row].append(p)

		rows = []
		
		for g in patches_groups_by_rows:
			newlist = sorted(patches_groups_by_rows[g], key=lambda x: x.gps.Center[0], reverse=False)
			
			rows.append(newlist)

		print('Rows calculated and created completely.')

		return rows

	def create_patches_SIFT_files(self):

		args_list = []

		for group in self.groups:
			for patch in group.patches:
				args_list.append(patch)
		
		processes = multiprocessing.Pool(no_of_cores_to_use)
		processes.map(parallel_patch_creator,args_list)
		processes.close()

	def save_plot(self):
		global plot_npy_file

		# result = []
		# color = 0

		# r = 0
		# g = 0
		# b = 0

		# for group in self.groups:
			
		# 	if color == 0:
		# 		color = 1
		# 	else:
		# 		color = 0

		# 	r+=10
		# 	g = 0
		# 	b = 0

		# 	for row in group.rows:
		# 		g+=10
		# 		b=0
		# 		for p in row:
		# 			b+=5
		# 			result.append([p.gps.Center[0],p.gps.Center[1],r,g,b])
		
		# np.save(plot_npy_file,np.array(result))	

		result = []
		color = 0

		for group in self.groups:
			
			if color == 0:
				color = 1
			else:
				color = 0

			for row in group.rows:
				
				for p in row:
					result.append([p.gps.Center[0],p.gps.Center[1],color])
		
		np.save(plot_npy_file,np.array(result))	

	def correct_groups_internally(self):
		global no_of_cores_to_use

		args_list = []

		for group in self.groups:

			args_list.append((group,1))

		processes = MyPool(int(no_of_cores_to_use))
		result = processes.map(correct_groups_internally_helper,args_list)
		processes.close()

		for r in result:
			
			string_res = r[0]

			gid = r[1]
			result_dict = get_result_dict_from_strings(string_res)

			for group in self.groups:
				
				if group.group_id == gid:

					for patch in group.patches:
						
						patch.gps = result_dict[patch.name]

		# manager = multiprocessing.Manager()
		# return_dict = manager.dict()
		# jobs = []

		# for group in self.groups:
			
		# 	p = multiprocessing.Process(target=correct_groups_internally_helper, args=(group.group_id,group,return_dict))
		# 	jobs.append(p)
		# 	p.daemon = False
		# 	p.start()		

		# for proc in jobs:
		# 	proc.join()

		# for i in return_dict:
		# 	string_res = return_dict[i]
		# 	result_dict = get_result_dict_from_strings(string_res)

		# 	for group in self.groups:
				
		# 		if group.group_id == i:

		# 			for patch in group.patches:
						
		# 				patch.gps = result_dict[patch.name]

	def correct_field(self):
		
		self.correct_groups_internally()

		print('Internally correction is finished.')
		sys.stdout.flush()

		previous_group = None

		for group in self.groups:
			
			if previous_group is None:
				# group.load_all_patches_SIFT_points()				
				previous_group = group
				continue

			# group.load_all_patches_SIFT_points()
			group.correct_self_based_on_previous_group(previous_group)
			# previous_group.delete_all_patches_SIFT_points()

			previous_group = group

		print('Field fully corrected.')
		sys.stdout.flush()

	def draw_and_save_field(self):
		global patch_folder, field_image_path

		all_patches = []

		for group in self.groups:

			all_patches+=[p for p in group.patches if (p not in all_patches)]

		up = all_patches[0].gps.UL_coord[1]
		down = all_patches[0].gps.LL_coord[1]
		left = all_patches[0].gps.UL_coord[0]
		right = all_patches[0].gps.UR_coord[0]

		for p in all_patches:
			if p.gps.UL_coord[1]>=up:
				up=p.gps.UL_coord[1]

			if p.gps.LL_coord[1]<=down:
				down=p.gps.LL_coord[1]

			if p.gps.UL_coord[0]<=left:
				left=p.gps.UL_coord[0]

			if p.gps.UR_coord[0]>=right:
				right=p.gps.UR_coord[0]


		super_patch_size = (int(math.ceil((up-down)/GPS_TO_IMAGE_RATIO[1]))+100,int(math.ceil((right-left)/GPS_TO_IMAGE_RATIO[0]))+100,3)
		UL = (left,up)

		result = np.zeros(super_patch_size)

		for p in all_patches:
			p.load_img()
			
			x_diff = p.gps.UL_coord[0] - UL[0]
			y_diff = UL[1] - p.gps.UL_coord[1]
			
			st_x = int(x_diff/GPS_TO_IMAGE_RATIO[0])
			st_y = int(y_diff/GPS_TO_IMAGE_RATIO[1])
			
			result[st_y:st_y+PATCH_SIZE[0],st_x:st_x+PATCH_SIZE[1],:] = p.rgb_img
			
			p.delete_img()

		result = cv2.resize(result,(int(result.shape[1]/10),int(result.shape[0]/10)))
		cv2.imwrite(field_image_path,result)
		print('Field successfully printed.')
		sys.stdout.flush()

	def save_new_coordinate(self):
		global CORRECTED_coordinates_file

		all_patches = []

		for group in self.groups:

			all_patches+=[p for p in group.patches if (p not in all_patches)]

		final_results = 'Filename,Upper left,Lower left,Upper right,Lower right,Center\n'

		for p in all_patches:
			p.gps.UL_coord = (round(p.gps.UL_coord[0],7),round(p.gps.UL_coord[1],7))
			p.gps.LL_coord = (round(p.gps.LL_coord[0],7),round(p.gps.LL_coord[1],7))
			p.gps.UR_coord = (round(p.gps.UR_coord[0],7),round(p.gps.UR_coord[1],7))
			p.gps.LR_coord = (round(p.gps.LR_coord[0],7),round(p.gps.LR_coord[1],7))
			p.gps.Center = (round(p.gps.Center[0],7),round(p.gps.Center[1],7))

			final_results += '{:s},"{:.7f},{:.7f}","{:.7f},{:.7f}","{:.7f},{:.7f}","{:.7f},{:.7f}","{:.7f},{:.7f}"\n'\
			.format(p.name,p.gps.UL_coord[0],p.gps.UL_coord[1],p.gps.LL_coord[0],p.gps.LL_coord[1],p.gps.UR_coord[0],p.gps.UR_coord[1]\
				,p.gps.LR_coord[0],p.gps.LR_coord[1],p.gps.Center[0],p.gps.Center[1])

		final_results = final_results.replace('(','"').replace(')','"')

		with open(CORRECTED_coordinates_file,'w') as f:
			f.write(final_results)

		print('Coordinates saved.')
		sys.stdout.flush()

def logger(corrected_patch,gps_diff,param,gid,step_id):
	global correction_log_file

	with open(correction_log_file,"a+") as f:
		

		if param.H is not None:
			string_log = '{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13}'.format(gid,step_id,corrected_patch.name,corrected_patch.gps.to_csv(),\
			param.H[0,2],param.H[1,2],param.num_matches,param.percentage_inliers,param.dissimilarity,gps_diff[0],gps_diff[1],\
			(param.overlap_on_patch[2]-param.overlap_on_patch[0])*(param.overlap_on_patch[3]-param.overlap_on_patch[1]),param.degrees,param.scale)
		else:
			string_log = '{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13}'.format(gid,step_id,corrected_patch.name,corrected_patch.gps.to_csv(),\
			None,None,param.num_matches,param.percentage_inliers,param.dissimilarity,gps_diff[0],gps_diff[1],\
			None,param.degrees,param.scale)

		f.write(string_log)


def test_function():
	global patch_folder

	images = os.listdir(patch_folder)
	img,img_g = load_preprocess_image('{0}/{1}'.format(patch_folder,images[0]))

	kp,desc = detect_SIFT_key_points(img_g,0,0,img_g.shape[1],img_g.shape[0])

	img_g=cv2.drawKeypoints(img_g,kp,img_g)

	cv2.namedWindow('fig1',cv2.WINDOW_NORMAL)
	cv2.namedWindow('fig2',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('fig1', 500,500)
	cv2.resizeWindow('fig2', 500,500)
	cv2.imshow('fig1',img)
	cv2.imshow('fig2',img_g)
	cv2.waitKey(0)

def main(scan_date):
	global server,patch_folder,SIFT_folder,lid_file,coordinates_file,CORRECTED_coordinates_file,plot_npy_file,row_save_path,field_image_path,lettuce_heads_coordinates_file,lettuce_coords,method,correction_log_file

	if server == 'coge':
		patch_folder = '/storage/ariyanzarei/{0}-rgb/bin2tif_out'.format(scan_date)
		SIFT_folder = '/storage/ariyanzarei/{0}-rgb/SIFT'.format(scan_date)
		lid_file = '/storage/ariyanzarei/{0}-rgb/lids.txt'.format(scan_date)
		coordinates_file = '/storage/ariyanzarei/{0}-rgb/{0}_coordinates.csv'.format(scan_date)
		CORRECTED_coordinates_file = '/storage/ariyanzarei/{0}-rgb/{0}_coordinates_CORRECTED.csv'.format(scan_date)
		plot_npy_file = '/storage/ariyanzarei/{0}-rgb/plt.npy'.format(scan_date)
		row_save_path = '/storage/ariyanzarei/{0}-rgb/rows'.format(scan_date)
		field_image_path = 'field.bmp'
		correction_log_file = '/storage/ariyanzarei/{0}-rgb/logs/log_{1}_at_{2}.csv'.format(scan_date,method,datetime.datetime.now().strftime("%d-%m-%y_%H:%M"))
		lettuce_heads_coordinates_file = 'season10_ind_lettuce_2020-05-27.csv'.format(scan_date)

	elif server == 'laplace.cs.arizona.edu':
		patch_folder = '/data/plant/full_scans/{0}-rgb/bin2tif_out'.format(scan_date)
		SIFT_folder = '/data/plant/full_scans/{0}-rgb/SIFT'.format(scan_date)
		lid_file = '/data/plant/full_scans/{0}-rgb/lids.txt'.format(scan_date)
		coordinates_file = '/data/plant/full_scans/metadata/{0}_coordinates.csv'.format(scan_date)
		CORRECTED_coordinates_file = '/data/plant/full_scans/metadata/{0}_coordinates_CORRECTED.csv'.format(scan_date)
		plot_npy_file = '/data/plant/full_scans/{0}-rgb/plt.npy'.format(scan_date)
		field_image_path = 'field.bmp'
		lettuce_heads_coordinates_file = 'season10_ind_lettuce_2020-05-27.csv'
		correction_log_file = '/data/plant/full_scans/{0}-rgb/logs/log_{1}_at_{2}.csv'.format(scan_date,method,datetime.datetime.now().strftime("%d-%m-%y_%H:%M"))

	elif server == 'ariyan':
		patch_folder = '/home/ariyan/Desktop/200203_Mosaic_Training_Data/200203_Mosaic_Training_Data/Figures'
		SIFT_folder = '/home/ariyan/Desktop/200203_Mosaic_Training_Data/200203_Mosaic_Training_Data/SIFT'
		lid_file = '/home/ariyan/Desktop/200203_Mosaic_Training_Data/200203_Mosaic_Training_Data/lids.txt'
		coordinates_file = '/home/ariyan/Desktop/200203_Mosaic_Training_Data/200203_Mosaic_Training_Data/coords.txt'
		CORRECTED_coordinates_file = '/home/ariyan/Desktop/200203_Mosaic_Training_Data/200203_Mosaic_Training_Data/coords2.txt'
		plot_npy_file = '/home/ariyan/Desktop/plt.npy'
		field_image_path = '/home/ariyan/Desktop/field.bmp'
		lettuce_heads_coordinates_file = '/home/ariyan/Desktop/season10_lettuce_latlon.csv'
		correction_log_file = ''

	else:
		# HPC
		patch_folder = '/xdisk/ericlyons/big_data/ariyanzarei/test_datasets/{0}-rgb/bin2tif_out'.format(scan_date)
		SIFT_folder = '/xdisk/ericlyons/big_data/ariyanzarei/test_datasets/{0}-rgb/SIFT'.format(scan_date)
		lid_file = '/xdisk/ericlyons/big_data/ariyanzarei/test_datasets/{0}-rgb/lids.txt'.format(scan_date)
		coordinates_file = '/xdisk/ericlyons/big_data/ariyanzarei/test_datasets/{0}-rgb/{1}_coordinates.csv'.format(scan_date,scan_date)
		CORRECTED_coordinates_file = '/xdisk/ericlyons/big_data/ariyanzarei/test_datasets/{0}-rgb/{1}_coordinates_CORRECTED.csv'.format(scan_date,scan_date)
		plot_npy_file = '/xdisk/ericlyons/big_data/ariyanzarei/test_datasets/{0}-rgb/plt.npy'.format(scan_date)
		field_image_path = '/xdisk/ericlyons/big_data/ariyanzarei/test_datasets/{0}-rgb/field.bmp'.format(scan_date)
		lettuce_heads_coordinates_file = '/xdisk/ericlyons/big_data/ariyanzarei/test_datasets/{0}-rgb/season10_ind_lettuce_2020-05-27.csv'.format(scan_date)
		correction_log_file = '/xdisk/ericlyons/big_data/ariyanzarei/test_datasets/{0}-rgb/logs/log_{1}_at_{2}.csv'.format(scan_date,method,datetime.datetime.now().strftime("%d-%m-%y_%H:%M"))


	if server == 'coge':
		print('RUNNING ON -- {0} --'.format(server))
		
		# field = Field()
		# lettuce_coords = read_lettuce_heads_coordinates()

		# field.save_plot()
		# field.create_patches_SIFT_files()

		# field.groups[14].correct_internally()
		# field.draw_and_save_field()
		# field.correct_field()
		# field.draw_and_save_field()
		# field.save_new_coordinate()
		print(calculate_error_of_correction())

	elif server == 'laplace.cs.arizona.edu':
		print('RUNNING ON -- {0} --'.format(server))
		os.system("taskset -p -c 0-37 %d" % os.getpid())
		
		# print(calculate_error_of_correction())

		# test_function()

		field = Field()
		# field.create_patches_SIFT_files()

		lettuce_coords = read_lettuce_heads_coordinates()
		# p1 = field.groups[0].patches[3]
		# p1.get_lettuce_contours_centers(lettuce_coords)
		# p1.correct_based_on_contours_and_lettuce_heads(lettuce_coords)

		# r = Row(field.groups[0].rows[0])

		# draw_together(field.groups[0].patches)
		# field.draw_and_save_field()
		# field.correct_field()
		# field.groups[0].load_all_patches_SIFT_points()
		# new_patches = super_patch_pool_merging_method(field.groups[0].patches)
		# field.draw_and_save_field()
		# r.correct_row_by_matching_lettuce_contours()
		# draw_together(new_patches)

		# correct_patch_group_all_corrected_neighbors(field.groups[0].patches)
		# print(calculate_error_of_correction(True))
		# field.draw_and_save_field()
		# field.groups[0].correct_internally()
		field.correct_field()
		# field.groups[0].correct_internally()
		# field.draw_and_save_field()
		field.save_new_coordinate()
		print(calculate_error_of_correction())

	elif server == 'ariyan':
		print('RUNNING ON -- {0} --'.format(server))

		# visualize_plot()

		patches = read_all_data()
		p1 = patches[0]
		
		# lettuce_coords = read_lettuce_heads_coordinates()
		# p1.get_lettuce_contours_centers(lettuce_coords)
		
		# fft = p1.get_fft_region(0,0,PATCH_SIZE[1],PATCH_SIZE[0])
		# # print(fft)
		# p1.load_img()

		p2 = patches[1]

		for p in patches:
			if p.has_overlap(p1) and p1.has_overlap(p) and p1 != p:
				p2 = p

				break
				
		
		p1.load_SIFT_points()
		p2.load_SIFT_points()
		p1.load_img()
		p2.load_img()
		overlap_1,overlap_2 = p1.get_overlap_rectangles(p2)

		# kp1,desc1 = choose_SIFT_key_points(p1,overlap_1[0],overlap_1[1],overlap_1[2],overlap_1[3])
		# kp2,desc2 = choose_SIFT_key_points(p2,overlap_2[0],overlap_2[1],overlap_2[2],overlap_2[3])
		kp1,desc1 = detect_SIFT_key_points(p1.rgb_img,overlap_1[0],overlap_1[1],overlap_1[2],overlap_1[3])
		kp2,desc2 = detect_SIFT_key_points(p2.rgb_img,overlap_2[0],overlap_2[1],overlap_2[2],overlap_2[3])

		matches = get_good_matches(desc1,desc2)
		sorted_matches = sorted(matches, key=lambda x: x[0].distance)

		img_top_lowerror = None
		img_top_higherror = None
		img_nottop_lowerror = None
		img_nottop_high_error = None

		mtch_top_lowerror = []
		mtch_top_higherror = []
		mtch_nottop_lowerror = []
		mtch_nottop_high_error = []

		for i,m in enumerate(sorted_matches):
			pt1 = kp1[m[0].queryIdx].pt
			pt2 = kp2[m[0].trainIdx].pt
			sg_tr = np.array([[1,0,PATCH_SIZE[0]-abs(pt1[1]-pt2[1])],[0,1,PATCH_SIZE[1]-abs(pt1[0]-pt2[0])],[0,0,1]])
			diff = get_gps_diff_from_H(p2,p1,sg_tr)
			print(diff)
			print(GPS_ERROR_X)
			print(GPS_ERROR_Y)

			if i<PERCENTAGE_OF_GOOD_MATCHES_FOR_GROUP_WISE_CORRECTION*len(sorted_matches):
				if abs(diff[0])<=GPS_ERROR_X and abs(diff[1])<=GPS_ERROR_Y:
					mtch_top_lowerror.append(m)
				else:
					mtch_top_higherror.append(m)
			else:
				if abs(diff[0])<=GPS_ERROR_X and abs(diff[1])<=GPS_ERROR_Y:
					mtch_nottop_lowerror.append(m)
				else:
					mtch_nottop_high_error.append(m)

		img_top_lowerror = cv2.drawMatches(p1.rgb_img,kp1,p2.rgb_img,kp2,[m[0] for m in mtch_top_lowerror],img_top_lowerror,matchColor=(0,255,0))
		img_top_higherror = cv2.drawMatches(p1.rgb_img,kp1,p2.rgb_img,kp2,[m[0] for m in mtch_top_higherror],img_top_higherror,matchColor=(0,255,0))
		img_nottop_lowerror = cv2.drawMatches(p1.rgb_img,kp1,p2.rgb_img,kp2,[m[0] for m in mtch_nottop_lowerror],img_nottop_lowerror,matchColor=(0,255,0))
		img_nottop_high_error = cv2.drawMatches(p1.rgb_img,kp1,p2.rgb_img,kp2,[m[0] for m in mtch_nottop_high_error],img_nottop_high_error,matchColor=(0,255,0))

		cv2.namedWindow('img_top_lowerror',cv2.WINDOW_NORMAL)
		cv2.namedWindow('img_top_higherror',cv2.WINDOW_NORMAL)
		cv2.namedWindow('img_nottop_lowerror',cv2.WINDOW_NORMAL)
		cv2.namedWindow('img_nottop_high_error',cv2.WINDOW_NORMAL)
		cv2.resizeWindow('img_top_lowerror', 500,500)
		cv2.resizeWindow('img_top_higherror', 500,500)
		cv2.resizeWindow('img_nottop_lowerror', 500,500)
		cv2.resizeWindow('img_nottop_high_error', 500,500)
		cv2.imshow('img_top_lowerror',img_top_lowerror)
		cv2.imshow('img_top_higherror',img_top_higherror)
		cv2.imshow('img_nottop_lowerror',img_nottop_lowerror)
		cv2.imshow('img_nottop_high_error',img_nottop_high_error)
		cv2.waitKey(0)

		# get_top_percentage_of_matches_no_KNN(p1,p2,desc1,desc2,kp1,kp2)


		# contours1 = p1.get_lettuce_contours(overlap=overlap_1)
		# contours2 = p2.get_lettuce_contours(overlap=overlap_2)

		# pairs = []

		# for i,cnt1 in enumerate(contours1):
		# 	for j,cnt2 in enumerate(contours2):
		# 		scr = cv2.matchShapes(cnt1,cnt2,1,0.0)
				
		# 		pairs.append((i,j,scr))
		
		# sorted_pairs = sorted(pairs, key = lambda x:x[2])
		# used_i = []
		# used_j = []

		# for p in sorted_pairs:
		# 	if p[0] in used_i or p[1] in used_j:
		# 		continue

		# 	r = random.randint(0,256)
		# 	g = random.randint(0,256)
		# 	b = random.randint(0,256)

		# 	cv2.drawContours(p1.rgb_img, contours1, p[0], (b,g,r),10)
		# 	cv2.drawContours(p2.rgb_img, contours2, p[1], (b,g,r),10)

		# 	used_i.append(p[0])
		# 	used_j.append(p[1])




		# cv2.namedWindow('img1',cv2.WINDOW_NORMAL)
		# cv2.namedWindow('img2',cv2.WINDOW_NORMAL)
		# cv2.resizeWindow('img1', 500,500)
		# cv2.resizeWindow('img2', 500,500)
		# cv2.imshow('img1',p1.rgb_img)
		# cv2.imshow('img2',p2.rgb_img)
		# cv2.waitKey(0)

		# draw_together([p1,p2])

		# p2.gps = p2.correct_based_on_neighbors([p1])
		# p2.correct_based_on_matched_contour_centers(p1)

		# draw_together([p1,p2])

		# p1.load_SIFT_points()
		# p2.load_SIFT_points()
		# p1.load_img()
		# p2.load_img()
		# tx = 0
		# ty = 0

		# while True:
		# 	p1.load_img()
		# 	p2.load_img()

		# 	p2.gps = add_to_gps_coord(p2.gps,tx,ty)
		# 	overlap_1,overlap_2 = p1.get_overlap_rectangles(p2)

		# 	img1 = p1.rgb_img.copy()
		# 	img2 = p2.rgb_img.copy()
			
		# 	# fd1 = p1.get_hog_region(overlap_1[0],overlap_1[1],overlap_1[2],overlap_1[3])
		# 	# fd2 = p2.get_hog_region(overlap_2[0],overlap_2[1],overlap_2[2],overlap_2[3])

		# 	fft1 = p1.get_fft_region(overlap_1[0],overlap_1[1],overlap_1[2],overlap_1[3])
		# 	fft2 = p2.get_fft_region(overlap_2[0],overlap_2[1],overlap_2[2],overlap_2[3])

		# 	print(np.sqrt(np.sum((fft1-fft2)**2)/(fft1.shape[0]*fft1.shape[1]*fft1.shape[2])))

		# 	cv2.rectangle(img1,(overlap_1[0],overlap_1[1]),(overlap_1[2],overlap_1[3]),(0,0,255),20)
		# 	cv2.rectangle(img2,(overlap_2[0],overlap_2[1]),(overlap_2[2],overlap_2[3]),(0,0,255),20)

		# 	cv2.namedWindow('p1',cv2.WINDOW_NORMAL)
		# 	cv2.namedWindow('p2',cv2.WINDOW_NORMAL)
		# 	cv2.namedWindow('fft1',cv2.WINDOW_NORMAL)
		# 	cv2.namedWindow('fft2',cv2.WINDOW_NORMAL)
		# 	cv2.resizeWindow('p1', 500,500)
		# 	cv2.resizeWindow('p2', 500,500)
		# 	cv2.resizeWindow('fft1', 500,500)
		# 	cv2.resizeWindow('fft2', 500,500)
		# 	cv2.imshow('p1',img1)
		# 	cv2.imshow('p2',img2)
		# 	cv2.imshow('fft1',fft1)
		# 	cv2.imshow('fft2',fft2)
		# 	key_pressed = cv2.waitKey(0)

			
		# 	if key_pressed == ord('a'):
		# 		tx = -0.0000001
		# 		ty = 0
		# 	elif key_pressed == ord('d'):
		# 		tx = +0.0000001
		# 		ty = 0
		# 	elif key_pressed == ord('w'):
		# 		tx = 0
		# 		ty = 0.0000001
		# 	elif key_pressed == ord('s'):
		# 		tx = 0
		# 		ty = -0.0000001


		

		# p1.load_img()
		# p2.load_img()

		# cv2.namedWindow('p1',cv2.WINDOW_NORMAL)
		# cv2.namedWindow('p2',cv2.WINDOW_NORMAL)
		# cv2.resizeWindow('p1', 500,500)
		# cv2.resizeWindow('p2', 500,500)

		# cv2.imshow('p1',p1.rgb_img)
		# cv2.imshow('p2',p2.rgb_img)
		# cv2.waitKey(0)

	else:
		# HPC
		print('RUNNING ON -- {0} --'.format(server))
		field = Field()
		# field.create_patches_SIFT_files()
		# field.draw_and_save_field()
		field.correct_field()
		field.draw_and_save_field()


		







server_core = {'coge':10,'laplace.cs.arizona.edu':10,'ariyan':4}

server = socket.gethostname()
if server not in ['coge','laplace.cs.arizona.edu','ariyan']:
	no_of_cores_to_use = 5
else:
	no_of_cores_to_use = server_core[server]


method = 'Hybrid'

start_time = datetime.datetime.now()

# main('2020-02-18')
main('2020-01-08')
# main('2020-05-18')

end_time = datetime.datetime.now()

report_time(start_time,end_time)
