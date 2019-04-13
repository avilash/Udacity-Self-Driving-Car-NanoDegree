import os
import cv2	
import numpy as np
import matplotlib.image as mpimg

from utils import write_img_pair
from calib import calibrate, undistort
from binary import gen_binary_image

def warp_top_view(img):

	height, width = img.shape
	# The crop is chosed in such a way that it covers half a lane on either side
	src = np.float32([[543,455],[734,457],[111,634],[1161,634]])
	dst = np.float32([[0,0],[width,0],[0,height],[width,height]])

	M = cv2.getPerspectiveTransform(src, dst)
	M_inv = cv2.getPerspectiveTransform(dst, src)
	warped = cv2.warpPerspective(img, M, (width, height), flags=cv2.INTER_LINEAR)

	return M, M_inv, warped


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
	M, M_inv, warped = warp_top_view(binary_result)

	write_img_pair(binary_result, warped, "Binary Image", "Image after Perpective Transform", 
		os.path.join(output_dir, file_name + "_perspective_result.jpg"))