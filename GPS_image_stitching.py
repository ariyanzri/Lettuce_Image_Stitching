import numpy as np
import cv2
import matplotlib.pyplot as plt

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
		img_res = main_img
		img_res = cv2.drawKeypoints(img_res,kp_n,img_res)
		ratio = img_res.shape[0]/img_res.shape[1]
		cv2.rectangle(img_res,(x1,y1),(x2,y2),(0,0,255),10)
		img_res = cv2.resize(img_res, (500, int(500*ratio))) 
		cv2.imshow('fig {0}'.format(n),img_res)
		cv2.waitKey(0)	

	return kp_n,desc

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
	dst = np.array([[ov_2_on_1[0],ov_2_on_1[3]]]).reshape(-1,1,2)
	dst = np.append(dst,np.array([[ov_2_on_1[2],ov_2_on_1[1]]]).reshape(-1,1,2),axis=0)
	src = np.array([[ov_1_on_2[0],ov_1_on_2[3]]]).reshape(-1,1,2)
	src = np.append(src,np.array([[ov_1_on_2[2],ov_1_on_2[1]]]).reshape(-1,1,2),axis=0)

	if len(matches)>0:
		src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
		dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)

	dst = np.append(dst,np.array([[ov_2_on_1[0],ov_2_on_1[1]]]).reshape(-1,1,2),axis=0)
	dst = np.append(dst,np.array([[ov_2_on_1[2],ov_2_on_1[3]]]).reshape(-1,1,2),axis=0)
	src = np.append(src,np.array([[ov_1_on_2[0],ov_1_on_2[1]]]).reshape(-1,1,2),axis=0)
	src = np.append(src,np.array([[ov_1_on_2[2],ov_1_on_2[3]]]).reshape(-1,1,2),axis=0)
	

	# H, masked = cv2.findHomography(dst, src, cv2.RANSAC, 3)
	# src = np.pad(src,[(0,0),(0,0),(0,1)],constant_values=1)
	# dst = np.pad(dst,[(0,0),(0,0),(0,1)],constant_values=1)

	H, masked = cv2.estimateAffinePartial2D(dst, src, maxIters = 9000, confidence = 0.999, refineIters = 15)
	H = np.append(H,np.array([[0,0,1]]),axis=0)
	H[0:2,0:2] = np.array([[1,0],[0,1]])
	return H
	
def stitch(rgb_img1,rgb_img2,img1,img2,H,show=False,write_out=True):
	T = np.array([[1,0,rgb_img1.shape[1]],[0,1,rgb_img1.shape[0]],[0,0,1]])

	H = T.dot(H)

	dst = cv2.warpPerspective(rgb_img1, H,(rgb_img2.shape[1] + 2*rgb_img1.shape[1], rgb_img2.shape[0]+2*rgb_img1.shape[0]))
	dst[rgb_img1.shape[0]:rgb_img1.shape[0]+rgb_img2.shape[0], rgb_img1.shape[1]:rgb_img1.shape[1]+rgb_img2.shape[1]] = rgb_img2
	
	ratio = dst.shape[0]/dst.shape[1]
	dst2 = cv2.resize(dst, (750, int(750*ratio))) 
	
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

	return dst

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
	def __init__(self,name,rgb_img,img,coords):
		self.name = name
		self.rgb_img = rgb_img
		self.img = img
		self.GPS_coords = coords
		self.size = np.shape(img)
		self.used_in_alg = True

	def has_overlap(self,p):
		if self.GPS_coords.is_coord_inside(p.GPS_coords.UL_coord) or self.GPS_coords.is_coord_inside(p.GPS_coords.UR_coord) or\
			self.GPS_coords.is_coord_inside(p.GPS_coords.LL_coord) or self.GPS_coords.is_coord_inside(p.GPS_coords.LR_coord):
			return True
		else:
			return False

	def get_overlap_rectangle(self,patch,increase_size=False):
		p1_x = 0
		p1_y = 0
		p2_x = self.size[1]
		p2_y = self.size[0]

		detect_overlap = False

		if self.GPS_coords.is_coord_inside(patch.GPS_coords.UL_coord):
			detect_overlap = True
			p1_x = int(((patch.GPS_coords.UL_coord[0]-self.GPS_coords.UL_coord[0]) / (self.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0]))*self.size[1])
			p1_y = int(((patch.GPS_coords.UL_coord[1]-self.GPS_coords.UR_coord[1]) / (self.GPS_coords.LR_coord[1]-self.GPS_coords.UR_coord[1]))*self.size[0])
		
		if self.GPS_coords.is_coord_inside(patch.GPS_coords.LR_coord):
			detect_overlap = True
			p2_x = int(((patch.GPS_coords.LR_coord[0]-self.GPS_coords.UL_coord[0]) / (self.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0]))*self.size[1])
			p2_y = int(((patch.GPS_coords.LR_coord[1]-self.GPS_coords.UR_coord[1]) / (self.GPS_coords.LR_coord[1]-self.GPS_coords.UR_coord[1]))*self.size[0])

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
	
def draw_GPS_coords_on_patch(patch,coord):
	if patch.GPS_coords.is_coord_inside(coord):
		x = int(((coord[0]-patch.GPS_coords.UL_coord[0]) / (patch.GPS_coords.UR_coord[0]-patch.GPS_coords.UL_coord[0]))*patch.size[1])
		y = int(((coord[1]-patch.GPS_coords.UR_coord[1]) / (patch.GPS_coords.LR_coord[1]-patch.GPS_coords.UR_coord[1]))*patch.size[0])
		cv2.circle(patch.img,(x,y),20,(0,0,255),thickness=-1)

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

def stitch_complete(patches):
	patches_tmp = patches.copy()

	i = 0

	while len(patches_tmp)>1:
		p = patches_tmp.pop()

		if p.used_in_alg == False:
			break

		overlaps = [p_n for p_n in patches_tmp if (p.has_overlap(p_n) or p_n.has_overlap(p))]

		if len(overlaps) == 0:
			print('Type1 >>> No overlaps for {0}. push back...'.format(p.name))
			patches_tmp.insert(0,p)
			p.used_in_alg = False
			continue

		p2 = overlaps[0]

		ov_2_on_1 = p.get_overlap_rectangle(p2)
		ov_1_on_2 = p2.get_overlap_rectangle(p)

		if (ov_2_on_1[0] == 0 and ov_2_on_1[1] == 0 and ov_2_on_1[2] == 0 and ov_2_on_1[3] == 0)\
		or (ov_1_on_2[0] == 0 and ov_1_on_2[1] == 0 and ov_1_on_2[2] == 0 and ov_1_on_2[3] == 0):
			print('Type2 >>> No overlaps for {0}. push back...'.format(p.name))
			print(ov_2_on_1)
			print(ov_1_on_2)
			patches_tmp.insert(0,p)
			p.used_in_alg = False
			continue

		avg_overlap_1 = np.mean(p.img[ov_2_on_1[1]:ov_2_on_1[3],ov_2_on_1[0]:ov_2_on_1[2]])
		avg_overlap_2 = np.mean(p.img[ov_1_on_2[1]:ov_1_on_2[3],ov_1_on_2[0]:ov_1_on_2[2]])

		if avg_overlap_1 == 0 or avg_overlap_2 == 0:
			print('Type3 >>> No overlap for {0}. push back...'.format(p.name))
			patches_tmp.insert(0,p)
			p.used_in_alg = False
			continue

		p2.used_in_alg = True
		patches_tmp.remove(p2)
		
		kp1,desc1 = detect_SIFT_key_points(p.img,ov_2_on_1[0],ov_2_on_1[1],ov_2_on_1[2],ov_2_on_1[3],1)
		kp2,desc2 = detect_SIFT_key_points(p2.img,ov_1_on_2[0],ov_1_on_2[1],ov_1_on_2[2],ov_1_on_2[3],2)

		matches = get_good_matches(desc2,desc1)

		H = find_homography(matches,kp2,kp1,ov_2_on_1,ov_1_on_2)

		result = stitch(p.rgb_img,p2.rgb_img,p.img,p2.img,H)
		
		stitched_coords = find_stitched_coords(p.GPS_coords,p2.GPS_coords)

		new_patch = Patch('New_{0}'.format(i),result,convert_to_gray(result),stitched_coords)

		patches_tmp.append(new_patch)

		i+=1

	
	for p in patches_tmp:
		ratio = p.rgb_img.shape[0]/p.rgb_img.shape[1]
		img_res = cv2.resize(p.rgb_img, (700, int(700*ratio))) 
		cv2.imshow('fig',img_res)
		cv2.waitKey(0)	


# rgb_img1,img1 = load_preprocess_image('/home/ariyan/Desktop/200203_Mosaic_Training_Data/200203_Mosaic_Training_Data/Figures/1a9de2f7-e67e-4283-a5e8-16d694a2258a_right.tif')
# rgb_img2,img2 = load_preprocess_image('/home/ariyan/Desktop/200203_Mosaic_Training_Data/200203_Mosaic_Training_Data/Figures/1cb7e153-12b6-44f1-a834-720eca1117b3_right.tif')

# rgb_img1,img1 = load_preprocess_image('/home/ariyan/Desktop/a.png')
# rgb_img2,img2 = load_preprocess_image('/home/ariyan/Desktop/b.png')

# kp1,desc1 = detect_SIFT_key_points(img1,int(np.shape(img1)[1]/2)+240,0,np.shape(img1)[1],np.shape(img1)[0])
# kp2,desc2 = detect_SIFT_key_points(img2,0,0,int(np.shape(img2)[1]/2)-300,np.shape(img2)[0])
# kp1,desc1 = detect_SIFT_key_points(img1,0,0,np.shape(img1)[1],np.shape(img1)[0])
# kp2,desc2 = detect_SIFT_key_points(img2,0,0,np.shape(img2)[1],np.shape(img2)[0])

# matches = get_good_matches(desc1,desc2)
# imm = hconcat_resize_min([img1,img2])


# H = find_homography(matches,kp1,kp2)


# # for m in matches[:,0]:
# # 	cv2.line(imm,(int(kp1[m.queryIdx].pt[0]),int(kp1[m.queryIdx].pt[1])),(int(kp2[m.trainIdx].pt[0])+np.shape(img2)[1],int(kp2[m.trainIdx].pt[1])),(120,0,250))

# # cv2.imshow('fig2',imm)
# # cv2.waitKey(0)

# stitch(rgb_img2,rgb_img1,img2,img1,H)

patches = read_all_data()
stitch_complete(patches)

