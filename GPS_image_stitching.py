import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import math
import multiprocessing
import datetime
import sys

def convert_to_gray(img):
	
	coefficients = [-1,1,2] 
	m = np.array(coefficients).reshape((1,3))
	img_g = cv2.transform(img, m)

	return img_g

def adjust_gamma(image, gamma=1.0):

	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
	
	return cv2.LUT(image, table)

def load_preprocess_image(address):
	img = cv2.imread(address)
	# img = adjust_gamma(img,1.2)
	img = img.astype('uint8')
	img_g = convert_to_gray(img)

	return img, img_g

def choose_SIFT_key_points(patch,x1,y1,x2,y2):
	kp = []
	desc = []

	for i,k in enumerate(patch.Keypoints_location):
		if k.pt[0]>= x1 and k.pt[0]<=x2 and k.pt[1]>=y1 and k.pt[1]<=y2:
			kp.append(k)

			desc.append(list(np.array(patch.Keypoints_desc[i,:])))

	desc = np.array(desc)
	
	return kp,desc

def detect_SIFT_key_points(img,x1,y1,x2,y2,n,show=False):
	sift = cv2.xfeatures2d.SIFT_create()
	main_img = img.copy()
	img = img[y1:y2,x1:x2]
	kp,desc = sift.detectAndCompute(img,None)

	kp_n = []
	for k in kp:
		kp_n.append(cv2.KeyPoint(k.pt[0]+x1,k.pt[1]+y1,k.size))

	kp = kp_n

	if show:
		img_res = main_img.copy()
		img_res = cv2.drawKeypoints(img_res,kp_n,img_res)
		ratio = img_res.shape[0]/img_res.shape[1]
		cv2.rectangle(img_res,(x1,y1),(x2,y2),(0,0,255),20)
		img_res = cv2.resize(img_res, (500, int(500*ratio))) 
		cv2.imshow('fig {0}'.format(n),img_res)
		cv2.waitKey(0)	

	return kp_n,desc

def matched_distance(p1,p2):

	return math.sqrt(np.sum((p1-p2)**2))

def get_good_matches(desc1,desc2):
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(desc1,desc2, k=2)

	good = []
	for m in matches:
		if m[0].distance < 0.8*m[1].distance:
			good.append(m)
	matches = np.asarray(good)

	return matches

def find_homography_gps_only(ov_2_on_1,ov_1_on_2,p1,p2):	

	# dst = [[ov_2_on_1[0],ov_2_on_1[3]] , [ov_2_on_1[2],ov_2_on_1[1]] , [ov_2_on_1[0],ov_2_on_1[1]]]
	# src = [[ov_1_on_2[0],ov_1_on_2[3]] , [ov_1_on_2[2],ov_1_on_2[1]] , [ov_1_on_2[0],ov_1_on_2[1]]]
	
	# dst = np.float32(dst)
	# src = np.float32(src)
	
	# H = cv2.getAffineTransform(dst, src)
	
	# H = np.append(H,np.array([[0,0,1]]),axis=0)
	# H[0:2,0:2] = np.array([[1,0],[0,1]])
	# return H

	ratio = ((p1.GPS_coords.UR_coord[0]-p1.GPS_coords.UL_coord[0])/p1.size[1],-(p1.GPS_coords.UL_coord[1]-p1.GPS_coords.LL_coord[1])/p1.size[0])

	diff_GPS = ((p1.GPS_coords.UL_coord[0]-p2.GPS_coords.UL_coord[0])/ratio[0],(p1.GPS_coords.UL_coord[1]-p2.GPS_coords.UL_coord[1])/ratio[1])

	H = np.eye(3)
	H[0,2] = diff_GPS[0]
	H[1,2] = diff_GPS[1]
	# print(H)
	return H

def find_homography(matches,kp1,kp2,ov_2_on_1,ov_1_on_2,add_gps):	
	if add_gps:
		dst = np.array([[ov_2_on_1[0],ov_2_on_1[3]]]).reshape(-1,1,2)
		dst = np.append(dst,np.array([[ov_2_on_1[2],ov_2_on_1[1]]]).reshape(-1,1,2),axis=0)
		src = np.array([[ov_1_on_2[0],ov_1_on_2[3]]]).reshape(-1,1,2)
		src = np.append(src,np.array([[ov_1_on_2[2],ov_1_on_2[1]]]).reshape(-1,1,2),axis=0)
		dst = np.append(dst,np.array([[ov_2_on_1[0],ov_2_on_1[1]]]).reshape(-1,1,2),axis=0)
		dst = np.append(dst,np.array([[ov_2_on_1[2],ov_2_on_1[3]]]).reshape(-1,1,2),axis=0)
		src = np.append(src,np.array([[ov_1_on_2[0],ov_1_on_2[1]]]).reshape(-1,1,2),axis=0)
		src = np.append(src,np.array([[ov_1_on_2[2],ov_1_on_2[3]]]).reshape(-1,1,2),axis=0)
		
	if len(matches)>0:
		src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
		dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
	else:
		return None,0

	# H, masked = cv2.findHomography(dst, src, cv2.RANSAC, 3)
	# src = np.pad(src,[(0,0),(0,0),(0,1)],constant_values=1)
	# dst = np.pad(dst,[(0,0),(0,0),(0,1)],constant_values=1)

	H, masked = cv2.estimateAffinePartial2D(dst, src, maxIters = 9000, confidence = 0.999, refineIters = 15)
	
	H = np.append(H,np.array([[0,0,1]]),axis=0)
	H[0:2,0:2] = np.array([[1,0],[0,1]])
	return H,np.sum(masked)/len(masked)
	
def is_point_inside(p1_1,p1_2,p1_3,p1_4,p2):
	if p2[0]>=p1_1[0] and p2[0]<=p1_2[0] and p2[1]>=p1_1[1] and p2[1]<=p1_3[1]:
		return True
	else:
		return False

def get_overlapped_region(dst,rgb_img1,rgb_img2,H):
	c1 = [0,0,1]
	c2 = [rgb_img1.shape[1],0,1]
	c3 = [0,rgb_img1.shape[0],1]
	c4 = [rgb_img1.shape[1],rgb_img1.shape[0],1]

	c1 = H.dot(c1).astype(int)
	c2 = H.dot(c2).astype(int)
	c3 = H.dot(c3).astype(int)
	c4 = H.dot(c4).astype(int)

	c2_1 = [rgb_img1.shape[1],rgb_img1.shape[0],1]
	c2_2 = [rgb_img1.shape[1]+rgb_img2.shape[1],rgb_img1.shape[0],1]
	c2_3 = [rgb_img1.shape[1],rgb_img1.shape[0]+rgb_img2.shape[0],1]
	c2_4 = [rgb_img1.shape[1]+rgb_img2.shape[1],rgb_img1.shape[0]+rgb_img2.shape[0],1]
	
	p1_x = c2_1[0]
	p1_y = c2_1[1]
	p2_x = c2_4[0]
	p2_y = c2_4[1]

	# print(c2_1,c2_2,c2_3,c2_4)
	# print(c1,c2,c3,c4)

	if is_point_inside(c2_1,c2_2,c2_3,c2_4,c1):
		p1_x = c1[0]
		p1_y = c1[1]
		# print('UL of img1')

	if is_point_inside(c1,c2,c3,c4,c2_4):
		p1_x = c1[0]
		p1_y = c1[1]
		# print('LR of img2')
	
	if is_point_inside(c2_1,c2_2,c2_3,c2_4,c4):
		p2_y = c4[1]
		p2_x = c4[0]
		# print('LR of img1')

	if is_point_inside(c1,c2,c3,c4,c2_1):
		p2_y = c4[1]
		p2_x = c4[0]
		# print('UL of img2')

	if is_point_inside(c2_1,c2_2,c2_3,c2_4,c2):
		p2_x = c2[0]
		p1_y = c2[1]
		# print('UR of img1')

	if is_point_inside(c1,c2,c3,c4,c2_3):
		p2_x = c2[0]
		p1_y = c2[1]
		# print('LL of img2')

	if is_point_inside(c2_1,c2_2,c2_3,c2_4,c3):
		p1_x = c3[0]
		p2_y = c3[1]
		# print('LL of img1')

	if is_point_inside(c1,c2,c3,c4,c2_2):
		p1_x = c3[0]
		p2_y = c3[1]
		# print('UR of img2')

	# cv2.rectangle(dst,(p1_x,p1_y),(p2_x,p2_y),(0,0,255),20)

	return np.copy(dst[p1_y:p2_y,p1_x:p2_x]),(p1_x,p1_y,p2_x,p2_y)

def revise_homography(H,rgb_img1,rgb_img2,img1,img2,move_steps,mse,length_rev):
	main_H = np.copy(H)
	min_H = np.copy(H)
	min_MSE = mse

	for r in range(0,length_rev):
		i = random.randint(-move_steps,move_steps+1)
		j = random.randint(-move_steps,move_steps+1)

		H = np.copy(main_H)
		H[0,2]+=i
		H[1,2]+=j

		dst = cv2.warpPerspective(rgb_img1, H,(rgb_img2.shape[1] + 2*rgb_img1.shape[1], rgb_img2.shape[0]+2*rgb_img1.shape[0]))
		overlap1,pnts = get_overlapped_region(dst,rgb_img1,rgb_img2,H)

		dst[rgb_img1.shape[0]:rgb_img1.shape[0]+rgb_img2.shape[0], rgb_img1.shape[1]:rgb_img1.shape[1]+rgb_img2.shape[1]] = rgb_img2
		overlap2,pnts = get_overlapped_region(dst,rgb_img1,rgb_img2,H)
		
		MSE = np.mean((overlap1-overlap2)**2)

		if MSE<min_MSE:
			min_MSE=MSE
			min_H = H 	


	return min_H

def stitch(rgb_img1,rgb_img2,img1,img2,H,overlap,show=False,write_out=False,apply_average=False,revise_h=False,revise_move_steps=20,length_rev=100):
	T = np.array([[1,0,rgb_img1.shape[1]],[0,1,rgb_img1.shape[0]],[0,0,1]])

	H = T.dot(H)
	# H = np.eye(3)

	# rgb_img2[overlap[1]:overlap[3],overlap[0]:overlap[2],:] = 0

	dst = cv2.warpPerspective(rgb_img1, H,(rgb_img2.shape[1] + 2*rgb_img1.shape[1], rgb_img2.shape[0]+2*rgb_img1.shape[0]))
	overlap1,pnts = get_overlapped_region(dst,rgb_img1,rgb_img2,H)

	dst[rgb_img1.shape[0]:rgb_img1.shape[0]+rgb_img2.shape[0], rgb_img1.shape[1]:rgb_img1.shape[1]+rgb_img2.shape[1]] = rgb_img2
	overlap2,pnts = get_overlapped_region(dst,rgb_img1,rgb_img2,H)

	mse_overlap1 = overlap1.copy()
	mse_overlap1[overlap1==0] = overlap2[overlap1==0]
	mse_overlap2 = overlap2.copy()
	mse_overlap2[overlap2==0] = overlap1[overlap2==0]

	MSE = np.mean((mse_overlap1-mse_overlap2)**2)

	del mse_overlap1
	del mse_overlap2

	if revise_h:
		H = revise_homography(H,rgb_img1,rgb_img2,img1,img2,revise_move_steps,MSE,length_rev)

		dst = cv2.warpPerspective(rgb_img1, H,(rgb_img2.shape[1] + 2*rgb_img1.shape[1], rgb_img2.shape[0]+2*rgb_img1.shape[0]))
		overlap1,pnts = get_overlapped_region(dst,rgb_img1,rgb_img2,H)

		dst[rgb_img1.shape[0]:rgb_img1.shape[0]+rgb_img2.shape[0], rgb_img1.shape[1]:rgb_img1.shape[1]+rgb_img2.shape[1]] = rgb_img2
		overlap2,pnts = get_overlapped_region(dst,rgb_img1,rgb_img2,H)
		
		mse_overlap1 = overlap1.copy()
		mse_overlap1[overlap1==0] = overlap2[overlap1==0]
		mse_overlap2 = overlap2.copy()
		mse_overlap2[overlap2==0] = overlap1[overlap2==0]
		
		MSE = np.mean((mse_overlap1-mse_overlap2)**2)
		
		del mse_overlap1
		del mse_overlap2

	overlap1[overlap1==0]=overlap2[overlap1==0]
	overlap2[overlap2==0]=overlap1[overlap2==0]
	
	dst[pnts[1]:pnts[3],pnts[0]:pnts[2]] = overlap2

	if apply_average:

		final_average = ((overlap1+overlap2)/2).astype('uint8')
		dst[pnts[1]:pnts[3],pnts[0]:pnts[2]] = final_average

	ratio = dst.shape[0]/dst.shape[1]
	dst2 = cv2.resize(dst, (1300, int(1300*ratio))) 
	
	print('\tMSE: {0}'.format(MSE))

	gray = convert_to_gray(dst2)
	coords = cv2.findNonZero(gray) 
	x, y, w, h = cv2.boundingRect(coords) 
	dst2 = dst2[y:y+h, x:x+w,:]

	gray = convert_to_gray(dst)
	coords = cv2.findNonZero(gray) 
	x, y, w, h = cv2.boundingRect(coords) 
	dst = dst[y:y+h, x:x+w,:]
	
	if show:
		cv2.imshow('fig',dst2)
		cv2.waitKey(0)

	if write_out:
		cv2.imwrite('output.jpg',dst)

	return dst,MSE

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

class Patch_GPS_coordinate:
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

	def __str__(self):
		return '---------------------------\nUL:{0}\nUR:{1}\nLL:{2}\nLR:{3}\n---------------------------\n'.format(self.UL_coord,self.UR_coord,self.LL_coord,self.LR_coord)
	
class Patch:

	def __init__(self,name,rgb_img,img,coords,kp=None,desc=None,size=None):
		self.name = name
		self.rgb_img = rgb_img
		self.img = img
		self.GPS_coords = coords
		if size == None:
			self.size = np.shape(img)
		else:
			self.size = size
		self.GPS_Corrected = False
		self.Keypoints_location = kp
		self.Keypoints_desc = desc

	def has_overlap(self,p):
		if self.GPS_coords.is_coord_inside(p.GPS_coords.UL_coord) or self.GPS_coords.is_coord_inside(p.GPS_coords.UR_coord) or\
			self.GPS_coords.is_coord_inside(p.GPS_coords.LL_coord) or self.GPS_coords.is_coord_inside(p.GPS_coords.LR_coord):
			return True
		else:
			return False

	def get_overlap_rectangle(self,patch,increase_size=True):
		p1_x = 0
		p1_y = 0
		p2_x = self.size[1]
		p2_y = self.size[0]

		detect_overlap = False

		# if self.GPS_coords.is_coord_inside(patch.GPS_coords.UL_coord):
		# 	detect_overlap = True
		# 	p1_x = int(((patch.GPS_coords.UL_coord[0]-self.GPS_coords.UL_coord[0]) / (self.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0]))*self.size[1])
		# 	p1_y = int(((patch.GPS_coords.UL_coord[1]-self.GPS_coords.UL_coord[1]) / (self.GPS_coords.LL_coord[1]-self.GPS_coords.UL_coord[1]))*self.size[0])
		
		# if self.GPS_coords.is_coord_inside(patch.GPS_coords.LR_coord):
		# 	detect_overlap = True
		# 	p2_x = int(((patch.GPS_coords.LR_coord[0]-self.GPS_coords.UL_coord[0]) / (self.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0]))*self.size[1])
		# 	p2_y = int(((patch.GPS_coords.LR_coord[1]-self.GPS_coords.UL_coord[1]) / (self.GPS_coords.LL_coord[1]-self.GPS_coords.UL_coord[1]))*self.size[0])

		# if self.GPS_coords.is_coord_inside(patch.GPS_coords.UR_coord):
		# 	detect_overlap = True
		# 	p2_x = int(((patch.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0]) / (self.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0]))*self.size[1])
		# 	p1_y = int(((patch.GPS_coords.UR_coord[1]-self.GPS_coords.UL_coord[1]) / (self.GPS_coords.LL_coord[1]-self.GPS_coords.UL_coord[1]))*self.size[0])

		# if self.GPS_coords.is_coord_inside(patch.GPS_coords.LL_coord):
		# 	detect_overlap = True
		# 	p1_x = int(((patch.GPS_coords.LL_coord[0]-self.GPS_coords.UL_coord[0]) / (self.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0]))*self.size[1])
		# 	p2_y = int(((patch.GPS_coords.LL_coord[1]-self.GPS_coords.UL_coord[1]) / (self.GPS_coords.LL_coord[1]-self.GPS_coords.UL_coord[1]))*self.size[0])

		if patch.GPS_coords.UL_coord[1]>=self.GPS_coords.LL_coord[1] and patch.GPS_coords.UL_coord[1]<=self.GPS_coords.UL_coord[1]:
			detect_overlap = True
			# print(patch.name+' upper border is inside')
			# p1_x = int(((patch.GPS_coords.UL_coord[0]-self.GPS_coords.UL_coord[0]) / (self.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0]))*self.size[1])
			p1_y = int(((patch.GPS_coords.UL_coord[1]-self.GPS_coords.UL_coord[1]) / (self.GPS_coords.LL_coord[1]-self.GPS_coords.UL_coord[1]))*self.size[0])
		
		if patch.GPS_coords.LL_coord[1]>=self.GPS_coords.LL_coord[1] and patch.GPS_coords.LL_coord[1]<=self.GPS_coords.UL_coord[1]:
			detect_overlap = True
			# print(patch.name+' lower border is inside')
			# p2_x = int(((patch.GPS_coords.LR_coord[0]-self.GPS_coords.UL_coord[0]) / (self.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0]))*self.size[1])
			p2_y = int(((patch.GPS_coords.LR_coord[1]-self.GPS_coords.UL_coord[1]) / (self.GPS_coords.LL_coord[1]-self.GPS_coords.UL_coord[1]))*self.size[0])

		if patch.GPS_coords.UR_coord[0]<=self.GPS_coords.UR_coord[0] and patch.GPS_coords.UR_coord[0]>=self.GPS_coords.UL_coord[0]:
			detect_overlap = True
			# print(patch.name+' right border is inside')
			p2_x = int(((patch.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0]) / (self.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0]))*self.size[1])
			# p1_y = int(((patch.GPS_coords.UR_coord[1]-self.GPS_coords.UL_coord[1]) / (self.GPS_coords.LL_coord[1]-self.GPS_coords.UL_coord[1]))*self.size[0])

		if patch.GPS_coords.UL_coord[0]<=self.GPS_coords.UR_coord[0] and patch.GPS_coords.UL_coord[0]>=self.GPS_coords.UL_coord[0]:
			detect_overlap = True
			# print(patch.name+' left border is inside')
			p1_x = int(((patch.GPS_coords.LL_coord[0]-self.GPS_coords.UL_coord[0]) / (self.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0]))*self.size[1])
			# p2_y = int(((patch.GPS_coords.LL_coord[1]-self.GPS_coords.UL_coord[1]) / (self.GPS_coords.LL_coord[1]-self.GPS_coords.UL_coord[1]))*self.size[0])

		if patch.GPS_coords.is_coord_inside(self.GPS_coords.UL_coord) and patch.GPS_coords.is_coord_inside(self.GPS_coords.UR_coord) and \
		patch.GPS_coords.is_coord_inside(self.GPS_coords.LL_coord) and patch.GPS_coords.is_coord_inside(self.GPS_coords.LR_coord):
			p1_x = 0
			p1_y = 0
			p2_x = self.size[1]
			p2_y = self.size[0]
			detect_overlap = True

		if increase_size:
			if p1_x>0+self.size[1]/10:
				p1_x-=self.size[1]/10

			if p2_x<9*self.size[1]/10:
				p2_x+=self.size[1]/10

			if p1_y>0+self.size[0]/10:
				p1_y-=self.size[0]/10

			if p2_y<9*self.size[0]/10:
				p2_y+=self.size[0]/10

		if detect_overlap == False:
			return 0,0,0,0

		return int(p1_x),int(p1_y),int(p2_x),int(p2_y)

def read_all_data():

	patches = []

	with open('/home/ariyan/Desktop/200203_Mosaic_Training_Data/200203_Mosaic_Training_Data/files.txt') as f:
		lines = f.read()
		for l in lines.split('\n'):
			if l == '':
				break

			rgb,img = load_preprocess_image('/home/ariyan/Desktop/200203_Mosaic_Training_Data/200203_Mosaic_Training_Data/Figures/{0}.tif'.format(l))
			patches.append(Patch(l,rgb,img,None))
	
	with open('/home/ariyan/Desktop/200203_Mosaic_Training_Data/200203_Mosaic_Training_Data/coords.txt') as f:
		lines = f.read()
		for l in lines.split('\n'):
			if l == '':
				break

			features = l.split(',')

			patch = [p for p in patches if p.name == features[0]]
			if len(patch) != 0:
				coord = Patch_GPS_coordinate((float(features[1].split(';')[0]),float(features[1].split(';')[1])),\
					(float(features[3].split(';')[0]),float(features[3].split(';')[1])),\
					(float(features[2].split(';')[0]),float(features[2].split(';')[1])),\
					(float(features[4].split(';')[0]),float(features[4].split(';')[1])),\
					(float(features[5].split(';')[0]),float(features[5].split(';')[1])))

				patch[0].GPS_coords = coord

	return patches

def parallel_patch_creator(address,filename,coord):
	rgb,img = load_preprocess_image('{0}/{1}'.format(address,filename))
	kp,desc = detect_SIFT_key_points(img,0,0,img.shape[1],img.shape[0],filename,False)
	kp_tmp = [(p.pt, p.size, p.angle, p.response, p.octave, p.class_id) for p in kp]     

	print('Patch created and SIFT generated for {0}'.format(filename))
	size = np.shape(img)
	p = Patch(filename,None,None,coord,kp_tmp,desc,size)
	del img
	del rgb

	return p

def parallel_patch_creator_helper(args):

	return parallel_patch_creator(*args)

def read_all_data_on_server(patches_address,metadatafile_address):

	patches = []

	with open(metadatafile_address) as f:
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

			print('{0}/{1}'.format(patches_address,filename))
			rgb,img = load_preprocess_image('{0}/{1}'.format(patches_address,filename))
			kp,desc = detect_SIFT_key_points(img,0,0,img.shape[1],img.shape[0],filename,False)
			

			coord = Patch_GPS_coordinate(upper_left,upper_right,lower_left,lower_right,center)

			patch = Patch(filename,None,None,coord,kp,desc,np.shape(img))
			
			patches.append(patch)

	return patches

	# ----------------- parallelism SIFT detecting --------------------------

	
	# args_list = []

	# with open(metadatafile_address) as f:
	# 	lines = f.read()
	# 	lines = lines.replace('"','')

	# 	for l in lines.split('\n'):
	# 		if l == '':
	# 			break
	# 		if l == 'Filename,Upper left,Lower left,Upper right,Lower right,Center' or l == 'name,upperleft,lowerleft,uperright,lowerright,center':
	# 			continue

	# 		features = l.split(',')

	# 		filename = features[0]
	# 		upper_left = (float(features[1]),float(features[2]))
	# 		lower_left = (float(features[3]),float(features[4]))
	# 		upper_right = (float(features[5]),float(features[6]))
	# 		lower_right = (float(features[7]),float(features[8]))
	# 		center = (float(features[9]),float(features[10]))

	# 		coord = Patch_GPS_coordinate(upper_left,upper_right,lower_left,lower_right,center)
			
	# 		args_list.append((patches_address,filename,coord))
			

	# 	processes = multiprocessing.Pool(4)
	# 	results = processes.map(parallel_patch_creator_helper,args_list)

	# 	for r in results:
	# 		tmp_kp = [cv2.KeyPoint(x=p[0][0],y=p[0][1],_size=p[1], _angle=p[2],_response=p[3], _octave=p[4], _class_id=p[5]) for p in r.Keypoints_location] 
	# 		r.Keypoints_location = tmp_kp


	# return results
	
def draw_GPS_coords_on_patch(patch,coord1,coord2,coord3,coord4):
	img = patch.rgb_img.copy()

	if patch.GPS_coords.is_coord_inside(coord1):
		
		x = int(((coord1[0]-patch.GPS_coords.UL_coord[0]) / (patch.GPS_coords.UR_coord[0]-patch.GPS_coords.UL_coord[0]))*patch.size[1])
		y = int(((coord1[1]-patch.GPS_coords.UR_coord[1]) / (patch.GPS_coords.LR_coord[1]-patch.GPS_coords.UR_coord[1]))*patch.size[0])
		cv2.circle(img,(x,y),100,(0,0,255),thickness=-1)
	if patch.GPS_coords.is_coord_inside(coord2):
		
		x = int(((coord2[0]-patch.GPS_coords.UL_coord[0]) / (patch.GPS_coords.UR_coord[0]-patch.GPS_coords.UL_coord[0]))*patch.size[1])
		y = int(((coord2[1]-patch.GPS_coords.UR_coord[1]) / (patch.GPS_coords.LR_coord[1]-patch.GPS_coords.UR_coord[1]))*patch.size[0])
		cv2.circle(img,(x,y),100,(0,255,0),thickness=-1)
	if patch.GPS_coords.is_coord_inside(coord3):
		
		x = int(((coord3[0]-patch.GPS_coords.UL_coord[0]) / (patch.GPS_coords.UR_coord[0]-patch.GPS_coords.UL_coord[0]))*patch.size[1])
		y = int(((coord3[1]-patch.GPS_coords.UR_coord[1]) / (patch.GPS_coords.LR_coord[1]-patch.GPS_coords.UR_coord[1]))*patch.size[0])
		cv2.circle(img,(x,y),100,(255,0,0),thickness=-1)
	if patch.GPS_coords.is_coord_inside(coord4):
		
		x = int(((coord4[0]-patch.GPS_coords.UL_coord[0]) / (patch.GPS_coords.UR_coord[0]-patch.GPS_coords.UL_coord[0]))*patch.size[1])
		y = int(((coord4[1]-patch.GPS_coords.UR_coord[1]) / (patch.GPS_coords.LR_coord[1]-patch.GPS_coords.UR_coord[1]))*patch.size[0])
		cv2.circle(img,(x,y),100,(255,0,255),thickness=-1)

	return img

def get_orientation(p1,p2,o1,o2):

	if o1[0]==0 and o1[1]==0 and o1[3]==p1.size[0]:
		# return '2 on left of 1'
		return 0
	if o2[0]==0 and o2[1]==0 and o2[3]==p2.size[0]:
		# return '1 on left of 2'
		return 1

	if o1[0]==0 and o1[1]==0 and o1[2]<p1.size[1]:
		# return '2 over 1 to the left'
		return 2
	if o1[0]>0 and o1[1]==0 and o1[2]==p1.size[1]:
		# return '2 over 1 to the right'
		return 3
	if o1[0]==0 and o1[1]==0 and o1[2]==p1.size[1]:
		# return '2 over 1'
		return 4
	
	if o2[0]==0 and o2[1]==0 and o2[2]<p2.size[1]:
		# return '1 over 2 to the left'
		return 5
	if o2[0]>0 and o2[1]==0 and o2[2]==p2.size[1]:
		# return '1 over 2 to the right'
		return 6
	if o2[0]==0 and o2[1]==0 and o2[2]==p2.size[1]:
		# return '1 over 2'
		return 7


	return 'else'

def find_stitched_coords(coord1,coord2):
	min_dim_0 = min(coord1.UL_coord[0],coord1.UR_coord[0],coord2.UL_coord[0],coord2.UR_coord[0])
	max_dim_0 = max(coord1.UL_coord[0],coord1.UR_coord[0],coord2.UL_coord[0],coord2.UR_coord[0])
	min_dim_1 = min(coord1.UL_coord[1],coord1.LL_coord[1],coord2.UL_coord[1],coord2.LL_coord[1])
	max_dim_1 = max(coord1.UL_coord[1],coord1.LL_coord[1],coord2.UL_coord[1],coord2.LL_coord[1])

	coord = Patch_GPS_coordinate((min_dim_0,max_dim_1),(max_dim_0,max_dim_1),(min_dim_0,min_dim_1),(max_dim_0,min_dim_1),((min_dim_0+max_dim_0)/2,(min_dim_1+max_dim_1)/2))
	
	return coord

def choose_patch_with_largest_overlap(patches):
	max_f = 0
	max_p = None

	for p in patches:
		new_patches=[ptmp for ptmp in patches if ptmp!=p]
		overlaps = [p_n for p_n in new_patches if (p.has_overlap(p_n) or p_n.has_overlap(p))]
		p_grbg ,f = get_best_overlap(p,overlaps)

		if f>max_f:
			maxf = f
			max_p = p

	return max_p

def get_best_overlap(p,overlaps,increase_size=True):
	max_f = -1
	max_p = None

	for p_ov in overlaps:
		rect_on_1 = p.get_overlap_rectangle(p_ov,increase_size)
		rect_on_2 = p_ov.get_overlap_rectangle(p,increase_size)
		f = (rect_on_1[2]-rect_on_1[0])*(rect_on_1[3]-rect_on_1[1])
		# f += abs(np.mean(p.img[rect_on_1[1]:rect_on_1[3],rect_on_1[0]:rect_on_1[2]])-np.mean(p_ov.img[rect_on_2[1]:rect_on_2[3],rect_on_2[0]:rect_on_2[2]]))*-1
		if f>max_f:
			max_f = f
			max_p = p_ov

	return max_p,max_f

def Get_GPS_Error(H,ov_1_on_2,ov_2_on_1):
	p2 = np.array([[ov_1_on_2[0],ov_1_on_2[1]]])
	p1 = np.array([[ov_2_on_1[0],ov_2_on_1[1]]])
	
	p1_translated = H.dot([p1[0,0],p1[0,1],1])

	# return int(matched_distance(p2,np.array([p1_translated[0],p1_translated[1]])))
	return int(abs(p2[0,0]-p1_translated[0])),int(abs(p2[0,1]-p1_translated[1]))

def get_new_GPS_Coords(p1,p2,H):

	rgb_img1 = p1.rgb_img
	rgb_img2 = p2.rgb_img

	c1 = [0,0,1]
	
	c1 = H.dot(c1).astype(int)
	
	# print(c1)

	diff_x = -c1[0]
	diff_y = -c1[1]

	gps_scale_x = (p2.GPS_coords.UR_coord[0] - p2.GPS_coords.UL_coord[0])/(p2.size[1])
	gps_scale_y = (p2.GPS_coords.LL_coord[1] - p2.GPS_coords.UL_coord[1])/(p2.size[0])

	diff_x = diff_x*gps_scale_x
	diff_y = diff_y*gps_scale_y

	# print(diff_x,diff_y)
	# print(p1.GPS_coords.UL_coord)
	new_UL = (round(p2.GPS_coords.UL_coord[0]-diff_x,7),round(p2.GPS_coords.UL_coord[1]-diff_y,7))
	# print(new_UL)
	diff_UL = (p1.GPS_coords.UL_coord[0]-new_UL[0],p1.GPS_coords.UL_coord[1]-new_UL[1])

	new_UR = (p1.GPS_coords.UR_coord[0]-diff_UL[0],p1.GPS_coords.UR_coord[1]-diff_UL[1])
	new_LL = (p1.GPS_coords.LL_coord[0]-diff_UL[0],p1.GPS_coords.LL_coord[1]-diff_UL[1])
	new_LR = (p1.GPS_coords.LR_coord[0]-diff_UL[0],p1.GPS_coords.LR_coord[1]-diff_UL[1])
	new_center = (p1.GPS_coords.Center[0]-diff_UL[0],p1.GPS_coords.Center[1]-diff_UL[1])

	new_coords = Patch_GPS_coordinate(new_UL,new_UR,new_LL,new_LR,new_center)

	return new_coords

def stitch_complete(patches,show,show2):
	patches_tmp = patches.copy()

	i = 0

	len_no_change = 0

	while len(patches_tmp)>1 and len_no_change<len(patches_tmp):
		p = patches_tmp.pop()
		len_no_change+=1

		# p = choose_patch_with_largest_overlap(patches_tmp)
		# patches_tmp.remove(p)

		overlaps = [p_n for p_n in patches_tmp if (p.has_overlap(p_n) or p_n.has_overlap(p))]

		if len(overlaps) == 0:
			print('Type1 >>> No overlaps for {0}. push back...'.format(p.name))
			patches_tmp.insert(0,p)
			continue

		p2,f_grbg = get_best_overlap(p,overlaps)
		# p2 = overlaps[0]
		
		ov_2_on_1 = p.get_overlap_rectangle(p2)
		ov_1_on_2 = p2.get_overlap_rectangle(p)
		
		if (ov_2_on_1[0] == 0 and ov_2_on_1[1] == 0 and ov_2_on_1[2] == 0 and ov_2_on_1[3] == 0)\
		or (ov_1_on_2[0] == 0 and ov_1_on_2[1] == 0 and ov_1_on_2[2] == 0 and ov_1_on_2[3] == 0):
			print('Type2 >>> Empty overlap for {0}. push back...'.format(p.name))
			patches_tmp.insert(0,p)
			
			# print(ov_2_on_1)
			# print(ov_1_on_2)
			# print(p.GPS_coords)
			# print(p2.GPS_coords)
			# cv2.namedWindow('fig1',cv2.WINDOW_NORMAL)
			# cv2.namedWindow('fig2',cv2.WINDOW_NORMAL)
			# cv2.resizeWindow('fig1', 600,600)
			# cv2.resizeWindow('fig2', 600,600)
			# cv2.imshow('fig1',p.rgb_img)
			# cv2.imshow('fig2',p2.rgb_img)
			# cv2.waitKey(0)
			continue

		avg_overlap_1 = np.sum(p.img[ov_2_on_1[1]:ov_2_on_1[3],ov_2_on_1[0]:ov_2_on_1[2]])
		avg_overlap_2 = np.sum(p.img[ov_1_on_2[1]:ov_1_on_2[3],ov_1_on_2[0]:ov_1_on_2[2]])

		# if avg_overlap_1 == 0 or avg_overlap_2 == 0:
		# 	print('Type3 >>> Blank overlap for {0}. push back...'.format(p.name))
		# 	patches_tmp.insert(0,p)
		# 	continue

		patches_tmp.remove(p2)
		
		kp1,desc1 = detect_SIFT_key_points(p.img,ov_2_on_1[0],ov_2_on_1[1],ov_2_on_1[2],ov_2_on_1[3],1,show)
		kp2,desc2 = detect_SIFT_key_points(p2.img,ov_1_on_2[0],ov_1_on_2[1],ov_1_on_2[2],ov_1_on_2[3],2,show)

		matches = get_good_matches(desc2,desc1)

		H,percentage_inliers = find_homography(matches,kp2,kp1,ov_2_on_1,ov_1_on_2,False)

		print('----Number of matches: {0}\n\tPercentage Iliers: {1}'.format(len(matches),percentage_inliers))

		if percentage_inliers<=0.1 or len(matches) < 40:
			patches_tmp.insert(0,p)
			patches_tmp.insert(0,p2)
			print('\t*** Not enough inliers ...')
			continue

		gps_err = Get_GPS_Error(H,ov_1_on_2,ov_2_on_1)
		
		if gps_err[0] > (ov_1_on_2[2]-ov_1_on_2[0])/2 and \
		gps_err[1] > (ov_1_on_2[3]-ov_1_on_2[1])/2:
			patches_tmp.insert(0,p)
			patches_tmp.insert(0,p2)
			print('*** High GPS error ...')
			continue


		result, MSE = stitch(p.rgb_img,p2.rgb_img,p.img,p2.img,H,ov_1_on_2,show)

		if MSE > 90:
			patches_tmp.insert(0,p)
			patches_tmp.insert(0,p2)
			print('\t*** High MSE ...')
			continue

		len_no_change = 0
		
		stitched_coords = find_stitched_coords(p.GPS_coords,p2.GPS_coords)

		new_patch = Patch('New_{0}'.format(i),result,convert_to_gray(result),stitched_coords)

		patches_tmp.append(new_patch)

		del p
		del p2

		i+=1

	print('############ Stitching based on GPS only ############')

	len_no_change = 0 

	while len(patches_tmp)>1 and len_no_change<len(patches_tmp):
		p = patches_tmp.pop()
		len_no_change+=1

		overlaps = [p_n for p_n in patches_tmp if (p.has_overlap(p_n) or p_n.has_overlap(p))]

		if len(overlaps) == 0:
			print('Type1 >>> No overlaps for {0}. push back...'.format(p.name))
			patches_tmp.insert(0,p)
			continue

		p2,f_grbg = get_best_overlap(p,overlaps,increase_size=False)
		# p2 = overlaps[0]
		
		ov_2_on_1 = p.get_overlap_rectangle(p2,increase_size=False)
		ov_1_on_2 = p2.get_overlap_rectangle(p,increase_size=False)
		
		if (ov_2_on_1[0] == 0 and ov_2_on_1[1] == 0 and ov_2_on_1[2] == 0 and ov_2_on_1[3] == 0)\
		or (ov_1_on_2[0] == 0 and ov_1_on_2[1] == 0 and ov_1_on_2[2] == 0 and ov_1_on_2[3] == 0):
			print('Type2 >>> Empty overlap for {0}. push back...'.format(p.name))
			patches_tmp.insert(0,p)
			
			# print(ov_2_on_1)
			# print(ov_1_on_2)
			# print(p.GPS_coords)
			# print(p2.GPS_coords)
			# cv2.namedWindow('fig1',cv2.WINDOW_NORMAL)
			# cv2.namedWindow('fig2',cv2.WINDOW_NORMAL)
			# cv2.resizeWindow('fig1', 600,600)
			# cv2.resizeWindow('fig2', 600,600)
			# img_res1 = p.rgb_img
			# img_res2 = p2.rgb_img
			# cv2.rectangle(img_res1,(ov_2_on_1[0],ov_2_on_1[1]),(ov_2_on_1[2],ov_2_on_1[3]),(0,0,255),20)
			# cv2.rectangle(img_res2,(ov_1_on_2[0],ov_1_on_2[1]),(ov_1_on_2[2],ov_1_on_2[3]),(0,0,255),20)
			# cv2.imshow('fig1',img_res1)
			# cv2.imshow('fig2',img_res2)
			# cv2.waitKey(0)
			continue

		avg_overlap_1 = np.sum(p.img[ov_2_on_1[1]:ov_2_on_1[3],ov_2_on_1[0]:ov_2_on_1[2]])
		avg_overlap_2 = np.sum(p.img[ov_1_on_2[1]:ov_1_on_2[3],ov_1_on_2[0]:ov_1_on_2[2]])

		patches_tmp.remove(p2)

		kp1,desc1 = detect_SIFT_key_points(p.img,ov_2_on_1[0],ov_2_on_1[1],ov_2_on_1[2],ov_2_on_1[3],1,show2)
		kp2,desc2 = detect_SIFT_key_points(p2.img,ov_1_on_2[0],ov_1_on_2[1],ov_1_on_2[2],ov_1_on_2[3],2,show2)

		matches = get_good_matches(desc2,desc1)

		H,percentage_inliers = find_homography(matches,kp2,kp1,ov_2_on_1,ov_1_on_2,True)
		
		print('----Number of matches: {0}\n\tPercentage Iliers: {1}'.format(len(matches),percentage_inliers))

		if percentage_inliers==0 or H is None:
			H = find_homography_gps_only(ov_2_on_1,ov_1_on_2)
		else:
			gps_err = Get_GPS_Error(H,ov_1_on_2,ov_2_on_1)
		
		if gps_err[0] > (ov_1_on_2[2]-ov_1_on_2[0])/2 and \
		gps_err[1] > (ov_1_on_2[3]-ov_1_on_2[1])/2:
			H = find_homography_gps_only(ov_2_on_1,ov_1_on_2)

		result, MSE = stitch(p.rgb_img,p2.rgb_img,p.img,p2.img,H,ov_1_on_2,show2)

		len_no_change = 0

		stitched_coords = find_stitched_coords(p.GPS_coords,p2.GPS_coords)

		new_patch = Patch('New_{0}'.format(i),result,convert_to_gray(result),stitched_coords)

		patches_tmp.append(new_patch)

		del p
		del p2

		i+=1

	return patches_tmp

def correct_GPS_coords(patches,show,show2):

	compute_neighbors(patches)

	patches_tmp = patches.copy()

	i = 0

	best_f = 0
	best_p = patches_tmp[0]

	for p in patches_tmp:
		overlaps = [p_n for p_n in patches_tmp if (p.has_overlap(p_n) or p_n.has_overlap(p))]
		f = len(overlaps)

		if f>best_f:
			best_f = f
			best_p = p

	best_p.GPS_Corrected = True

	while True:

		not_corrected_patches = [p for p in patches_tmp if p.GPS_Corrected == False]
		if len(not_corrected_patches) == 0:
			break

		p = not_corrected_patches[-1]
		patches_tmp.remove(p)

		overlaps = [p_n for p_n in patches_tmp if ((p_n.GPS_Corrected) and (p.has_overlap(p_n) or p_n.has_overlap(p)))]

		if len(overlaps) == 0:
			print('Type1 >>> No overlaps for {0}. push back...'.format(p.name))
			patches_tmp.insert(0,p)
			continue

		p2,f_grbg = get_best_overlap(p,overlaps)
		
		ov_2_on_1 = p.get_overlap_rectangle(p2)
		ov_1_on_2 = p2.get_overlap_rectangle(p)
		
		if (ov_2_on_1[0] == 0 and ov_2_on_1[1] == 0 and ov_2_on_1[2] == 0 and ov_2_on_1[3] == 0)\
		or (ov_1_on_2[0] == 0 and ov_1_on_2[1] == 0 and ov_1_on_2[2] == 0 and ov_1_on_2[3] == 0):
			print('Type2 >>> Empty overlap for {0}. push back...'.format(p.name))
			patches_tmp.insert(0,p)
			
			continue

		patches_tmp.insert(0,p)

		# kp1,desc1 = detect_SIFT_key_points(p.img,ov_2_on_1[0],ov_2_on_1[1],ov_2_on_1[2],ov_2_on_1[3],1,show)
		# kp2,desc2 = detect_SIFT_key_points(p2.img,ov_1_on_2[0],ov_1_on_2[1],ov_1_on_2[2],ov_1_on_2[3],2,show)

		kp1,desc1 = choose_SIFT_key_points(p,ov_2_on_1[0],ov_2_on_1[1],ov_2_on_1[2],ov_2_on_1[3])
		kp2,desc2 = choose_SIFT_key_points(p2,ov_1_on_2[0],ov_1_on_2[1],ov_1_on_2[2],ov_1_on_2[3])

		matches = get_good_matches(desc2,desc1)

		H,percentage_inliers = find_homography(matches,kp2,kp1,ov_2_on_1,ov_1_on_2,False)

		print('----Number of matches: {0}\n\tPercentage Iliers: {1}'.format(len(matches),percentage_inliers))

		if percentage_inliers<=0.1 or len(matches) < 40:
			patches_tmp.insert(0,p)
			patches_tmp.insert(0,p2)
			print('\t*** Not enough inliers ...')
			continue

		gps_err = Get_GPS_Error(H,ov_1_on_2,ov_2_on_1)
		
		if gps_err[0] > (ov_1_on_2[2]-ov_1_on_2[0])/2 and \
		gps_err[1] > (ov_1_on_2[3]-ov_1_on_2[1])/2:
			patches_tmp.insert(0,p)
			patches_tmp.insert(0,p2)
			print('*** High GPS error ...')
			continue

		if show2:
			G = find_homography_gps_only(ov_2_on_1,ov_1_on_2,p,p2)
			result, MSE = stitch(p.rgb_img,p2.rgb_img,p.img,p2.img,G,ov_1_on_2,show2)

		p.GPS_coords = get_new_GPS_Coords(p,p2,H)
		p.GPS_Corrected = True


		print('GPC corrected for {0}.'.format(p.name))

		if show2:
			ov_2_on_1 = p.get_overlap_rectangle(p2)
			ov_1_on_2 = p2.get_overlap_rectangle(p)

			G = find_homography_gps_only(ov_2_on_1,ov_1_on_2,p,p2)
			result, MSE = stitch(p.rgb_img,p2.rgb_img,p.img,p2.img,G,ov_1_on_2,show2)

	return patches_tmp

def stitch_based_on_corrected_GPS(patches,show):
	patches_tmp = patches.copy()

	i = 0

	len_no_change = 0

	while len(patches_tmp)>1 and len_no_change<len(patches_tmp):
		p = patches_tmp.pop()
		len_no_change+=1

		# p = choose_patch_with_largest_overlap(patches_tmp)
		# patches_tmp.remove(p)

		overlaps = [p_n for p_n in patches_tmp if (p.has_overlap(p_n) or p_n.has_overlap(p))]

		if len(overlaps) == 0:
			print('Type1 >>> No overlaps for {0}. push back...'.format(p.name))
			patches_tmp.insert(0,p)
			continue

		p2,f_grbg = get_best_overlap(p,overlaps)
		
		ov_2_on_1 = p.get_overlap_rectangle(p2)
		ov_1_on_2 = p2.get_overlap_rectangle(p)
		
		if (ov_2_on_1[0] == 0 and ov_2_on_1[1] == 0 and ov_2_on_1[2] == 0 and ov_2_on_1[3] == 0)\
		or (ov_1_on_2[0] == 0 and ov_1_on_2[1] == 0 and ov_1_on_2[2] == 0 and ov_1_on_2[3] == 0):
			print('Type2 >>> Empty overlap for {0}. push back...'.format(p.name))
			patches_tmp.insert(0,p)
			
			continue

		patches_tmp.remove(p2)
		
		H = find_homography_gps_only(ov_2_on_1,ov_1_on_2,p,p2)

		result, MSE = stitch(p.rgb_img,p2.rgb_img,p.img,p2.img,H,ov_1_on_2,show)

		len_no_change = 0
		
		stitched_coords = find_stitched_coords(p.GPS_coords,p2.GPS_coords)

		new_patch = Patch('New_{0}'.format(i),result,convert_to_gray(result),stitched_coords)

		patches_tmp.append(new_patch)

		del p
		del p2

		i+=1

	return patches_tmp

def show_and_save_final_patches(patches):
	for p in patches:
		cv2.imwrite('{0}.jpg'.format(p.name),p.rgb_img)
		ratio = p.rgb_img.shape[0]/p.rgb_img.shape[1]
		img_res = cv2.resize(p.rgb_img, (700, int(700*ratio))) 
		cv2.imshow('fig',img_res)
		cv2.waitKey(0)	

def save_coordinates(final_patches,filename):
	
	final_results = 'Filename,Upper left,Lower left,Upper right,Lower right,Center\n'

	for p in final_patches:
		p.GPS_coords.UL_coord = (round(p.GPS_coords.UL_coord[0],7),round(p.GPS_coords.UL_coord[1],7))
		p.GPS_coords.LL_coord = (round(p.GPS_coords.LL_coord[0],7),round(p.GPS_coords.LL_coord[1],7))
		p.GPS_coords.UR_coord = (round(p.GPS_coords.UR_coord[0],7),round(p.GPS_coords.UR_coord[1],7))
		p.GPS_coords.LR_coord = (round(p.GPS_coords.LR_coord[0],7),round(p.GPS_coords.LR_coord[1],7))
		p.GPS_coords.Center = (round(p.GPS_coords.Center[0],7),round(p.GPS_coords.Center[1],7))

		final_results += '{:s},"{:.7f},{:.7f}","{:.7f},{:.7f}","{:.7f},{:.7f}","{:.7f},{:.7f}","{:.7f},{:.7f}"\n'\
		.format(p.name,p.GPS_coords.UL_coord[0],p.GPS_coords.UL_coord[1],p.GPS_coords.LL_coord[0],p.GPS_coords.LL_coord[1],p.GPS_coords.UR_coord[0],p.GPS_coords.UR_coord[1]\
			,p.GPS_coords.LR_coord[0],p.GPS_coords.LR_coord[1],p.GPS_coords.Center[0],p.GPS_coords.Center[1])

	final_results = final_results.replace('(','"').replace(')','"')

	with open(filename,'w') as f:
		f.write(final_results)

def main():

	# patches = read_all_data_on_server('/home/ariyan/Desktop/200203_Mosaic_Training_Data/200203_Mosaic_Training_Data/Figures','/home/ariyan/Desktop/200203_Mosaic_Training_Data/200203_Mosaic_Training_Data/coords2.txt')
	# final_patches = stitch_complete(patches,True,True)
	# final_patches = correct_GPS_coords(patches,False,False)
	# final_patches = stitch_based_on_corrected_GPS(patches,True)
	# save_coordinates(final_patches,'/home/ariyan/Desktop/200203_Mosaic_Training_Data/200203_Mosaic_Training_Data/coords2.txt')
	# show_and_save_final_patches(final_patches)

	patches = read_all_data_on_server('/data/plant/full_scans/2020-01-08-rgb/bin2tif_out','/data/plant/full_scans/metadata/2020-01-08_coordinates.csv')
	final_patches = correct_GPS_coords(patches,False,False)
	save_coordinates(final_patches,'/data/plant/full_scans/metadata/2020-01-08_coordinates_CORRECTED.csv')

def report_time(start,end):
	print('Start date time: {0}\nEnd date time: {1}\nTotal running time: {2}.'.format(start,end,end-start))

start_time = datetime.datetime.now()
main()
end_time = datetime.datetime.now()
report_time(start_time,end_time)