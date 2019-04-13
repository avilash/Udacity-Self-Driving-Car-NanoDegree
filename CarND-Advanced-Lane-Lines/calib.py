import os
import cv2	
import numpy as np
import matplotlib.image as mpimg

from utils import write_img_pair

def load_img_obj_points(calib_dir):
	nx = 9
	ny = 6
	img_points = []
	object_points = [] 

	# Define the array of object points
	obj_point = np.zeros((nx * ny, 3), dtype=np.float32)
	for i in range(nx):
		for j in range(ny):
			obj_point[i*ny + j] = (i, j, 0)

	# Iterate through the images and find image points for all the images. 
	# Keep duplicating obejct points
	for f in os.listdir(calib_dir):
		img = cv2.imread(os.path.join(calib_dir, f))
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
		if ret:
			img_points.append(corners)
			object_points.append(obj_point)
		
	return img_points, object_points

def calibrate(calib_dir):
	img_points, object_points = load_img_obj_points(calib_dir)
	images = os.listdir(calib_dir)
	img = cv2.imread(os.path.join(calib_dir, images[0]))
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, img_points, gray.shape[::-1], None, None)
	return mtx, dist

def undistort(img, mtx, dist):
	return cv2.undistort(img, mtx, dist, None, mtx)

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
	
	write_img_pair(img, undistorted, "Original", "Undistorted", 
		os.path.join(output_dir, file_name + "_undist_result.jpg"))