import os
import cv2	
import numpy as np
import matplotlib.image as mpimg

from utils import write_img_pair
from calib import calibrate, undistort

def gen_binary_image(img, hls_thresh=([0, 70, 70], [50, 255, 255]), sx_thresh=(20, 100)):
	binary_result = np.zeros_like(img[:,:,0])
	
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	h_channel = hls[:,:,0]
	l_channel = hls[:,:,1]
	s_channel = hls[:,:,2]

	# Color thresholding
	color_binary = cv2.inRange(hls, np.array(hls_thresh[0]), np.array(hls_thresh[1]))
	color_binary[color_binary == 255] = 1

	# Apply Sobel gradient along X
	sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
	abs_sobelx = np.absolute(sobelx)
	scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
	sxbinary = np.zeros_like(scaled_sobel)
	sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

	# Combine results
	binary_result = np.logical_or(color_binary, sxbinary)
	binary_result = binary_result.astype(np.uint8)
	color_binary_result = np.dstack(( np.zeros_like(sxbinary), sxbinary, color_binary)) * 255

	return binary_result, color_binary_result

if __name__ =='__main__':

	calib_dir = 'camera_cal'
	test_img_dir = 'test_images'
	output_dir = 'output_images'

	test_images = os.listdir(test_img_dir)
	file_idx = 5
	file_name = os.path.splitext(test_images[file_idx])[0]
	img = mpimg.imread(os.path.join(test_img_dir, test_images[file_idx]))

	mtx, dist = calibrate(calib_dir)
	undistorted = undistort(img, mtx, dist)
	binary_result, color_binary_result = gen_binary_image(undistorted)
		
	write_img_pair(undistorted, color_binary_result, "Original Image", "Binary Image", 
		os.path.join(output_dir, file_name + "_binary_result.jpg"))