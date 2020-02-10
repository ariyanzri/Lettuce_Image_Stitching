import numpy as np
import cv2
import matplotlib.pyplot as plt

def convert_to_gray(img):
	
	coefficients = [0.2,0.7,0.1] 
	m = np.array(coefficients).reshape((1,3))
	res = cv2.transform(img, m)

	return res

def load_preprocess_image(address):
	img = cv2.imread(address)
	# img_g = convert_to_gray(img)
	img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	edge = cv2.Canny(img_g,40,80)
	img_g+=edge

	return img, img_g

def detect_SIFT_key_points(img,x1,y1,x2,y2):
	sift = cv2.xfeatures2d.SIFT_create()
	main_img = img.copy()

	img = img[y1:y2,x1:x2]

	kp,desc = sift.detectAndCompute(img,None)

	kp_n = []
	for k in kp:
		kp_n.append(cv2.KeyPoint(k.pt[0]+x1,k.pt[1]+y1,k.size))

	kp = kp_n
	img_res = main_img
	img_res = cv2.drawKeypoints(img_res,kp_n,img_res)
	ratio = img_res.shape[0]/img_res.shape[1]
	cv2.rectangle(img_res,(x1,y1),(x2,y2),(0,0,255),10)
	img_res = cv2.resize(img_res, (500, int(500*ratio))) 
	cv2.imshow('fig {0}'.format(x1),img_res)
	cv2.waitKey(0)	

	return kp_n,desc

def get_good_matches(desc1,desc2):
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(desc1,desc2, k=2)

	good = []
	for m in matches:
		if m[0].distance < 0.5*m[1].distance:
			good.append(m)
	matches = np.asarray(good)

	return matches

def find_homography(matches,kp1,kp2):
	if len(matches[:,0]) >= 4:
		src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
		dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
		H, masked = cv2.findHomography(dst, src, cv2.RANSAC, 3)

		return H
	else:
		return None
		
def stitch(rgb_img1,rgb_img2,img1,img2,H):
	dst = cv2.warpPerspective(rgb_img1, H,(rgb_img2.shape[1] + rgb_img1.shape[1], rgb_img2.shape[0]))
	dst[0:rgb_img2.shape[0], 0:rgb_img2.shape[1]] = rgb_img2
	cv2.imwrite('output.jpg',dst)
	ratio = dst.shape[0]/dst.shape[1]
	dst = cv2.resize(dst, (960, int(960*ratio))) 
	cv2.imshow('fig',dst)
	cv2.waitKey(0)

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

rgb_img1,img1 = load_preprocess_image('/home/ariyan/Desktop/200203_Mosaic_Training_Data/200203_Mosaic_Training_Data/Figures/1a9de2f7-e67e-4283-a5e8-16d694a2258a_right.tif')
rgb_img2,img2 = load_preprocess_image('/home/ariyan/Desktop/200203_Mosaic_Training_Data/200203_Mosaic_Training_Data/Figures/1cb7e153-12b6-44f1-a834-720eca1117b3_right.tif')

# rgb_img1,img1 = load_preprocess_image('/home/ariyan/Desktop/a.png')
# rgb_img2,img2 = load_preprocess_image('/home/ariyan/Desktop/b.png')

kp1,desc1 = detect_SIFT_key_points(img1,int(np.shape(img1)[1]/2)+240,0,np.shape(img1)[1],np.shape(img1)[0])
kp2,desc2 = detect_SIFT_key_points(img2,0,0,int(np.shape(img2)[1]/2)-300,np.shape(img2)[0])
# kp1,desc1 = detect_SIFT_key_points(img1,0,0,np.shape(img1)[1],np.shape(img1)[0])
# kp2,desc2 = detect_SIFT_key_points(img2,0,0,np.shape(img2)[1],np.shape(img2)[0])

matches = get_good_matches(desc1,desc2)
imm = hconcat_resize_min([img1,img2])


H = find_homography(matches,kp1,kp2)


# for m in matches[:,0]:
# 	cv2.line(imm,(int(kp1[m.queryIdx].pt[0]),int(kp1[m.queryIdx].pt[1])),(int(kp2[m.trainIdx].pt[0])+np.shape(img2)[1],int(kp2[m.trainIdx].pt[1])),(120,0,250))

# cv2.imshow('fig2',imm)
# cv2.waitKey(0)

stitch(rgb_img2,rgb_img1,img2,img1,H)