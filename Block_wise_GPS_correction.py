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
from heapq import heappush, heappop, heapify
from collections import OrderedDict

PATCH_SIZE = (3296, 2472)
PATCH_SIZE_GPS = (8.899999997424857e-06,1.0199999998405929e-05)
HEIGHT_RATIO_FOR_ROW_SEPARATION = 0.1
NUMBER_OF_ROWS_IN_GROUPS = 10
NUMBER_OF_GOOD_MATCHES_FOR_GROUP_WISE_CORRECTION = 3000
GPS_TO_IMAGE_RATIO = (PATCH_SIZE_GPS[0]/PATCH_SIZE[1],PATCH_SIZE_GPS[1]/PATCH_SIZE[0])


def convert_to_gray(img):
	
	coefficients = [-1,1,2] 
	m = np.array(coefficients).reshape((1,3))
	img_g = cv2.transform(img, m)

	return img_g

def load_preprocess_image(address):
	img = cv2.imread(address)
	
	img = img.astype('uint8')
	img_g = convert_to_gray(img)

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

def get_good_matches(desc1,desc2):
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(desc1,desc2, k=2)

	good = []
	for m in matches:
		if m[0].distance < 0.8*m[1].distance:
			good.append(m)
	matches = np.asarray(good)

	return matches

def find_homography(matches,kp1,kp2,ov_2_on_1,ov_1_on_2):	
	
	if len(matches)>1:
		src = np.float32([ kp1[m.queryIdx] for m in matches[:,0] ]).reshape(-1,1,2)
		dst = np.float32([ kp2[m.trainIdx] for m in matches[:,0] ]).reshape(-1,1,2)
	else:
		return None,0

	H, masked = cv2.estimateAffinePartial2D(dst, src, maxIters = 9000, confidence = 0.999, refineIters = 15)
	
	H = np.append(H,np.array([[0,0,1]]),axis=0)
	H[0:2,0:2] = np.array([[1,0],[0,1]])
	return H,np.sum(masked)/len(masked)

def get_dissimilarity_on_overlaps(p1,p2,H,ov1,ov2):

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

	return dissimilarity

def visualize_plot():
	global plot_npy_file
	import matplotlib.pyplot as plt

	plt.axis('equal')

	data = np.load(plot_npy_file)

	c = []
	for d in data:
		c.append((d[2]/255,d[3]/255,d[4]/255))

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

	return args[0].correct_internally()


def get_top_n_good_matches(desc1,desc2,kp1,kp2):
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(desc1,desc2, k=2)

	good = []
	for m in matches:
		
		if 	m[0].distance < 0.8*m[1].distance:
			good.append(m)

	sorted_matches = sorted(good, key=lambda x: x[0].distance)

	good = []

	if len(sorted_matches)>NUMBER_OF_GOOD_MATCHES_FOR_GROUP_WISE_CORRECTION:
		good += sorted_matches[0:NUMBER_OF_GOOD_MATCHES_FOR_GROUP_WISE_CORRECTION]
	else:
		good += sorted_matches

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


def get_result_dict_from_strings(strings):

	res_dict = {}

	for s in strings:
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

# --------------- new method in which we consider all patches -------------------

def find_all_neighbors(patches,patch):

	neighbors = []
	for p in patches:
		if (p.has_overlap(patch) or patch.has_overlap(p)) and p != patch:
			neighbors.append(p)
	return neighbors

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
	patch.load_img()

	for p in corrected_neighbors:
		p.load_img()
		p.load_SIFT_points()

		overlap = p.get_overlap_rectangle(patch)
		kp,desc = choose_SIFT_key_points(p,overlap[0],overlap[1],overlap[2],overlap[3])
		
		x_diff = p.gps.UL_coord[0] - UL[0]
		y_diff = UL[1] - p.gps.UL_coord[1]
		
		st_x = int(math.ceil(x_diff/GPS_TO_IMAGE_RATIO[0]))
		st_y = int(math.ceil(y_diff/GPS_TO_IMAGE_RATIO[1]))
		
		result[st_y:st_y+PATCH_SIZE[0],st_x:st_x+PATCH_SIZE[1],:] = p.rgb_img
		for i,k in enumerate(kp):
			total_kp.append((k[0]+st_x,k[1]+st_y))
			total_desc.append(desc[i,:])
			# cv2.circle(result,(k[0]+st_x,k[1]+st_y),2,(0,0,255),-1)

		p.delete_img()

	total_desc = np.array(total_desc)

	result = np.array(result).astype('uint8')
	result = cv2.resize(result,(int(result.shape[1]/5),int(result.shape[0]/5)))
	img = patch.rgb_img.copy()
	img = cv2.resize(img,(int(PATCH_SIZE[1]/5),int(PATCH_SIZE[0]/5)))
	cv2.imshow('figmain',img)
	cv2.imshow('fig',result)
	cv2.waitKey(0)

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

	new_UL = (UL[0]-diff_x,UL[1]-diff_y,7)

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
		if shape_1[0]<shape_2[0]:
			overlap_2_img = overlap_2_img[:shape_1[0],:,:]
			shape_2 = shape_1
		if shape_1[1]<shape_2[1]:
			overlap_2_img = overlap_2_img[:,:shape_1[1],:]
			shape_2 = shape_1
		
		if shape_2[0]<shape_1[0]:
			overlap_1_img = overlap_1_img[:shape_2[0],:,:]
			shape_1 = shape_2
		if shape_2[1]<shape_1[1]:
			overlap_1_img = overlap_1_img[:,:shape_2[1],:]
			shape_1 = shape_2

	if shape_1[0] == 0 or shape_1[1] == 0 or shape_2[0] == 0 or shape_2[1] == 0:
		return sys.maxsize

	# overlap_1_img = cv2.cvtColor(overlap_1_img, cv2.COLOR_BGR2GRAY)
	# overlap_2_img = cv2.cvtColor(overlap_2_img, cv2.COLOR_BGR2GRAY)

	overlap_1_img = cv2.blur(overlap_1_img,(5,5))
	overlap_2_img = cv2.blur(overlap_2_img,(5,5))

	# ret1,overlap_1_img = cv2.threshold(overlap_1_img,0,255,cv2.THRESH_OTSU)
	# ret1,overlap_2_img = cv2.threshold(overlap_2_img,0,255,cv2.THRESH_OTSU)

	# tmp_size = np.shape(overlap_1_img)
	
	# overlap_1_img[overlap_1_img==255] = 1
	# overlap_2_img[overlap_2_img==255] = 1

	# xnor_images = np.logical_xor(overlap_1_img,overlap_2_img)

	# dissimilarity = round(np.sum(xnor_images)/(tmp_size[0]*tmp_size[1]),2)
	dissimilarity =  np.sum((overlap_1_img.astype("float") - overlap_2_img.astype("float")) ** 2)
	dissimilarity /= float(overlap_1_img.shape[0] * overlap_1_img.shape[1]* overlap_1_img.shape[2])
	

	return dissimilarity

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
	
	list_jitter_x = [-0.0000004,-0.0000003,-0.0000002,-0.0000001,0,0.0000001,0.0000002,0.0000003,0.0000004]
	list_jitter_y = [-0.0000004,-0.0000003,-0.0000002,-0.0000001,0,0.0000001,0.0000002,0.0000003,0.0000004]

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

def correct_patch_group_all_corrected_neighbors(patches):
	corrected_patches = [patches[0]]
	can_be_corrected_patches = find_all_neighbors(patches,patches[0])

	while len(corrected_patches)<len(patches):
		patch = can_be_corrected_patches.pop()

		tmp_neighbors = find_all_neighbors(patches,patch)
		corrected_neighbors = [p for p in tmp_neighbors if p in corrected_patches]

		UL_merged, kp_merged, desc_merged = merge_all_neighbors(corrected_neighbors,patch)
		patch.load_SIFT_points()
		kp = patch.SIFT_kp_locations
		desc = patch.SIFT_kp_desc

		matches = get_good_matches(desc_merged,desc)

		H, perc_in = find_homography(matches,kp_merged,kp,None,None)

		coord = get_new_GPS_Coords_all_neighbors(patch,UL_merged,H)

		patch.gps = coord

		patch.gps = jitter_image_to_find_least_dissimilarity(patch,corrected_neighbors)

		corrected_patches.append(patch)
		can_be_corrected_patches=[t for t in tmp_neighbors if (t not in corrected_patches) and (t not in can_be_corrected_patches)]+can_be_corrected_patches
		# H = get_transformation_from_all_corrected_neighbors(patch,corrected_neighbors)


# ----------------------------------------------------------------------


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


class Graph():

	def __init__(self,no_vertex,vertex_names):
		self.vertecis_number = no_vertex
		self.vertex_index_to_name_dict = {}
		self.vertex_name_to_index_dict = {}
		for i,v in enumerate(vertex_names):
			self.vertex_index_to_name_dict[i] = v
			self.vertex_name_to_index_dict[v] = i

		self.edges = [[-1 for column in range(no_vertex)] for row in range(no_vertex)]

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
		min_value = 1

		for v in range(self.vertecis_number): 
			if keys[v] < min_value and mstSet[v] == False: 
				min_value = keys[v] 
				min_index = v 

		return min_index 


	def generate_MST_prim(self,starting_vertex):
		keys = [1]*self.vertecis_number
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

		while len(queue_traverse) > 0:
			u = queue_traverse.pop()

			for v,p in enumerate(parents):
				if p == u:
					queue_traverse = [v] + queue_traverse

					patch = dict_patches[self.vertex_index_to_name_dict[v]]
					parent_patch = dict_patches[self.vertex_index_to_name_dict[p]]
					H = [n[1].H for n in parent_patch.neighbors if n[0] == patch]
					H = H[0]

					patch.gps = get_new_GPS_Coords(patch,parent_patch,H)

		string_corrected = get_corrected_string(patches)
		return string_corrected

class Neighbor_Parameters:
	def __init__(self,o_p,o_n,h,nm,pi,d):

		self.overlap_on_patch = o_p
		self.overlap_on_neighbor = o_n
		self.H = h
		self.num_matches = nm
		self.percentage_inliers = pi
		self.dissimilarity = d


class Patch:
	
	def __init__(self,name,coord):
		self.name = name
		self.gps = coord
		self.neighbors = []
		self.SIFT_kp_locations = []
		self.SIFT_kp_desc = []

	def __eq__(self,other):

		return (self.name == other.name)

	def has_overlap(self,p):
		if self.gps.is_coord_inside(p.gps.UL_coord) or self.gps.is_coord_inside(p.gps.UR_coord) or\
			self.gps.is_coord_inside(p.gps.LL_coord) or self.gps.is_coord_inside(p.gps.LR_coord):
			return True
		else:
			return False

	def load_SIFT_points(self):
		global SIFT_folder

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

	def get_overlap_rectangle(self,patch,increase_size=True):
		p1_x = 0
		p1_y = 0
		p2_x = PATCH_SIZE[1]
		p2_y = PATCH_SIZE[0]

		detect_overlap = False

		if patch.gps.UL_coord[1]>=self.gps.LL_coord[1] and patch.gps.UL_coord[1]<=self.gps.UL_coord[1]:
			detect_overlap = True
			p1_y = int(math.ceil(((patch.gps.UL_coord[1]-self.gps.UL_coord[1]) / (self.gps.LL_coord[1]-self.gps.UL_coord[1]))*PATCH_SIZE[0]))
		
		if patch.gps.LL_coord[1]>=self.gps.LL_coord[1] and patch.gps.LL_coord[1]<=self.gps.UL_coord[1]:
			detect_overlap = True
			p2_y = int(math.ceil(((patch.gps.LR_coord[1]-self.gps.UL_coord[1]) / (self.gps.LL_coord[1]-self.gps.UL_coord[1]))*PATCH_SIZE[0]))

		if patch.gps.UR_coord[0]<=self.gps.UR_coord[0] and patch.gps.UR_coord[0]>=self.gps.UL_coord[0]:
			detect_overlap = True
			p2_x = int(math.ceil(((patch.gps.UR_coord[0]-self.gps.UL_coord[0]) / (self.gps.UR_coord[0]-self.gps.UL_coord[0]))*PATCH_SIZE[1]))
			
		if patch.gps.UL_coord[0]<=self.gps.UR_coord[0] and patch.gps.UL_coord[0]>=self.gps.UL_coord[0]:
			detect_overlap = True
			p1_x = int(math.ceil(((patch.gps.LL_coord[0]-self.gps.UL_coord[0]) / (self.gps.UR_coord[0]-self.gps.UL_coord[0]))*PATCH_SIZE[1]))
			
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

	def get_pairwise_transformation_info(self,neighbor):
		overlap1 = neighbor.get_overlap_rectangle(self)
		overlap2 = self.get_overlap_rectangle(neighbor)
		
		if overlap1[2]-overlap1[0]<PATCH_SIZE[1]/5 and overlap1[3]-overlap1[1]<PATCH_SIZE[0]/5:
			
			return None

		kp1,desc1 = choose_SIFT_key_points(neighbor,overlap1[0],overlap1[1],overlap1[2],overlap1[3])
		kp2,desc2 = choose_SIFT_key_points(self,overlap2[0],overlap2[1],overlap2[2],overlap2[3])

		matches = get_good_matches(desc2,desc1)

		num_matches = len(matches)

		H,percentage_inliers = find_homography(matches,kp2,kp1,overlap1,overlap2)

		if H is None:
			
			return None

		percentage_inliers = round(percentage_inliers*100,2)

		dissimilarity = get_dissimilarity_on_overlaps(neighbor,self,H,overlap1,overlap2)

		if dissimilarity == -1:
			
			return None

		return Neighbor_Parameters(overlap2,overlap1,H,num_matches,percentage_inliers,dissimilarity)


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

	def pre_calculate_internal_neighbors_and_transformation_parameters(self):
		remove_neighbors = []

		for p in self.patches:

			for n in self.patches:

				if n != p and (p.has_overlap(n) or n.has_overlap(p)):

					neighbor_param = p.get_pairwise_transformation_info(n)
					
					if neighbor_param == None:
						remove_neighbors.append((n,p))
						continue
					
					p.neighbors.append((n,neighbor_param))

			print('GROPU ID: {0} - Calculated Transformation and error values for {1} neighbors of {2}'.format(self.group_id,len(p.neighbors),p.name))
			sys.stdout.flush()

		for a,b in remove_neighbors:
			new_neighbors = []

			for n in a.neighbors:
				if b != n[0]:
					new_neighbors.append(n)
				
			a.neighbors = new_neighbors


	def correct_internally(self):

		self.load_all_patches_SIFT_points()

		self.pre_calculate_internal_neighbors_and_transformation_parameters()

		G = Graph(len(self.patches),[p.name for p in self.patches])
		G.initialize_edge_weights(self.patches)

		parents = G.generate_MST_prim(self.rows[0][0].name)
		string_res = G.revise_GPS_from_generated_MST(self.patches,parents)

		self.delete_all_patches_SIFT_points()

		print('Group {0} was corrected internally. '.format(self.group_id))
		sys.stdout.flush()

		return string_res


	def correct_self_based_on_previous_group(self,previous_group):

		matches = []
		kp = []
		desc = []
		prev_kp = []
		prev_desc = []

		for self_patch in self.rows[0]:
			
			for prev_patch in previous_group.rows[-1]:

				if self_patch.has_overlap(prev_patch) or prev_patch.has_overlap(self_patch):
					overlap1 = self_patch.get_overlap_rectangle(prev_patch)
					overlap2 = prev_patch.get_overlap_rectangle(self_patch)

					kp1,desc1 = choose_SIFT_key_points(self_patch,overlap1[0],overlap1[1],overlap1[2],overlap1[3])
					kp2,desc2 = choose_SIFT_key_points(prev_patch,overlap2[0],overlap2[1],overlap2[2],overlap2[3])

					# print('overlap detected. {0}-\n\t{1}'.format(overlap1,overlap2))

					kp.append(kp1)
					desc.append(desc1)
					prev_kp.append(kp2)
					prev_desc.append(desc2)
					
					matches.append(get_top_n_good_matches(desc2,desc1,kp2,kp1))

		H = calculate_homography_for_super_patches(prev_kp,kp,matches)

		base_patch_from_prev = previous_group.rows[-1][0]

		for patch in self.patches:

			patch.gps = get_new_GPS_Coords_for_groups(patch,base_patch_from_prev,H)

		print('Block {0} corrected based on previous block.'.format(self.group_id))
		sys.stdout.flush()

class Field:
	def __init__(self):
		global coordinates_file

		self.groups = self.initialize_field()

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

	def save_plot(self):
		global plot_npy_file

		result = []
		color = 0

		r = 0
		g = 0
		b = 0

		for group in self.groups:
			
			if color == 0:
				color = 1
			else:
				color = 0

			r+=10
			g = 0
			b = 0

			for row in group.rows:
				g+=10
				b=0
				for p in row:
					b+=5
					result.append([p.gps.Center[0],p.gps.Center[1],r,g,b])
		
		np.save(plot_npy_file,np.array(result))	

	def correct_groups_internally(self):
		global no_of_cores_to_use

		args_list = []

		for group in self.groups:

			args_list.append((group,1))

		processes = multiprocessing.Pool(int(no_of_cores_to_use/2))
		str_results = processes.map(correct_groups_internally_helper,args_list)
		processes.close()

		result_dict = get_result_dict_from_strings(str_results)

		for group in self.groups:
			for patch in group.patches:
				patch.gps = result_dict[patch.name]


	def correct_field(self):
		
		self.correct_groups_internally()

		print('Internally correction is finished.')
		sys.stdout.flush()

		previous_group = None

		for group in self.groups:
			
			if previous_group is None:
				group.load_all_patches_SIFT_points()				
				previous_group = group
				continue

			group.load_all_patches_SIFT_points()
			group.correct_self_based_on_previous_group(previous_group)
			previous_group.delete_all_patches_SIFT_points()

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

def main():
	global server,patch_folder,SIFT_folder,lid_file,coordinates_file,CORRECTED_coordinates_file,plot_npy_file,row_save_path,field_image_path

	if server == 'coge':
		patch_folder = '/storage/ariyanzarei/2020-01-08-rgb/bin2tif_out'
		SIFT_folder = '/storage/ariyanzarei/2020-01-08-rgb/SIFT'
		lid_file = '/storage/ariyanzarei/2020-01-08-rgb/lids.txt'
		coordinates_file = '/storage/ariyanzarei/2020-01-08-rgb/2020-01-08_coordinates.csv'
		CORRECTED_coordinates_file = '/storage/ariyanzarei/2020-01-08-rgb/2020-01-08_coordinates_CORRECTED.csv'
		plot_npy_file = '/storage/ariyanzarei/2020-01-08-rgb/plt.npy'
		row_save_path = '/storage/ariyanzarei/2020-01-08-rgb/rows'
		field_image_path = 'field.bmp'

	elif server == 'laplace.cs.arizona.edu':
		patch_folder = '/data/plant/full_scans/2020-01-08-rgb/bin2tif_out'
		SIFT_folder = '/data/plant/full_scans/2020-01-08-rgb/SIFT'
		lid_file = '/data/plant/full_scans/2020-01-08-rgb/lids.txt'
		coordinates_file = '/data/plant/full_scans/metadata/2020-01-08_coordinates.csv'
		CORRECTED_coordinates_file = '/data/plant/full_scans/metadata/2020-01-08_coordinates_CORRECTED.csv'
		plot_npy_file = '/data/plant/full_scans/2020-01-08-rgb/plt.npy'
		field_image_path = 'field.bmp'

	elif server == 'ariyan':
		patch_folder = '/home/ariyan/Desktop/200203_Mosaic_Training_Data/200203_Mosaic_Training_Data/Figures'
		SIFT_folder = '/home/ariyan/Desktop/200203_Mosaic_Training_Data/200203_Mosaic_Training_Data/SIFT'
		lid_file = '/home/ariyan/Desktop/200203_Mosaic_Training_Data/200203_Mosaic_Training_Data/lids.txt'
		coordinates_file = '/home/ariyan/Desktop/200203_Mosaic_Training_Data/200203_Mosaic_Training_Data/coords.txt'
		CORRECTED_coordinates_file = '/home/ariyan/Desktop/200203_Mosaic_Training_Data/200203_Mosaic_Training_Data/coords2.txt'
		plot_npy_file = '/home/ariyan/Desktop/plt.npy'
		field_image_path = '/home/ariyan/Desktop/field.bmp'


	if server == 'coge':
		print('RUNNING ON -- {0} --'.format(server))
		
		field = Field()
		field.save_plot()
		# field.correct_field()
		# field.draw_and_save_field()
		# field.save_new_coordinate()

	elif server == 'laplace.cs.arizona.edu':
		print('RUNNING ON -- {0} --'.format(server))
		# os.system("taskset -p -c 1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23,24,25,27,28,29,30,31,32,33,34,35,36,37,38,39,44,45,46 %d" % os.getpid())
		
		field = Field()
		correct_patch_group_all_corrected_neighbors(field.groups[0].patches)

		# field.draw_and_save_field()
		# field.correct_field()
		# field.draw_and_save_field()
		# field.save_new_coordinate()

	elif server == 'ariyan':
		print('RUNNING ON -- {0} --'.format(server))
		visualize_plot()



server_core = {'coge':64,'laplace.cs.arizona.edu':30,'ariyan':4}

server = socket.gethostname()
no_of_cores_to_use = server_core[server]

start_time = datetime.datetime.now()
main()
end_time = datetime.datetime.now()
report_time(start_time,end_time)
