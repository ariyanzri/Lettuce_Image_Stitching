import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_preprocess_image(address):
	img = cv2.imread(address)
	img_g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	return img,img_g

def detect_SIFT_key_points(img):
	sift = cv2.xfeatures2d.SIFT_create()
	kp,desc = sift.detectAndCompute(img,None)

	# img_res = img.copy()
	# img_res = cv2.drawKeypoints(img,kp,img_res)
	# ratio = img.shape[0]/img.shape[1]
	# img_res = cv2.resize(img_res, (960, int(960*ratio))) 
	# cv2.imshow('fig',img_res)
	# cv2.waitKey(0)

	return kp,desc

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
		H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 0.01)

		return H
	else:
		return None
		
def stitch(rgb_img1,rgb_img2,img1,img2,H):
	dst = cv2.warpPerspective(rgb_img1,H,(rgb_img2.shape[1] + rgb_img1.shape[1], rgb_img2.shape[0]))
	dst[0:rgb_img2.shape[0], 0:rgb_img2.shape[1]] = rgb_img2
	cv2.imwrite('output.jpg',dst)
	ratio = dst.shape[0]/dst.shape[1]
	dst = cv2.resize(dst, (960, int(960*ratio))) 
	cv2.imshow('fig',dst)
	cv2.waitKey(0)

rgb_img1,img1 = load_preprocess_image('/home/ariyan/Desktop/200203_Mosaic_Training_Data/200203_Mosaic_Training_Data/Figures/1a9de2f7-e67e-4283-a5e8-16d694a2258a_right.tif')
rgb_img2,img2 = load_preprocess_image('/home/ariyan/Desktop/200203_Mosaic_Training_Data/200203_Mosaic_Training_Data/Figures/1cb7e153-12b6-44f1-a834-720eca1117b3_right.tif')

# rgb_img1,img1 = load_preprocess_image('/home/ariyan/Desktop/a.jpg')
# rgb_img2,img2 = load_preprocess_image('/home/ariyan/Desktop/b.jpg')

kp1,desc1 = detect_SIFT_key_points(img1)
kp2,desc2 = detect_SIFT_key_points(img2)

matches = get_good_matches(desc1,desc2)
H = find_homography(matches,kp1,kp2)

stitch(rgb_img1,rgb_img2,img1,img2,H)