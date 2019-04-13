import os
import cv2	
import argparse
import numpy as np
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

from utils import write_img_pair
from calib import calibrate, undistort
from binary import gen_binary_image
from perspective import warp_top_view
from lane import find_lane_curves, find_lane_curves_seed, draw_lane_curves_on_road, get_offset_from_center
from curve import Curve

def stitch_all_results(img, binary, warped, lane_curves, l_curvature, r_curvature, offset):
	h, w, d = img.shape
	scale = 0.20
	sh, sw = int(h*scale), int(w*scale)
	pad = 20

	binary *=255
	binary = cv2.resize(binary, (0,0), fx=scale, fy=scale)
	binary = np.dstack((binary, binary, binary)) 
	warped *=255
	warped = cv2.resize(warped, (0,0), fx=scale, fy=scale) 
	warped = np.dstack((warped, warped, warped)) 
	lane_curves = cv2.resize(lane_curves, (0,0), fx=scale, fy=scale)

	curr_x = pad
	img[pad:pad+sh, curr_x:curr_x+sw] = binary
	curr_x += sw + pad
	img[pad:pad+sh, curr_x:curr_x+sw] = warped
	curr_x += sw + pad
	img[pad:pad+sh, curr_x:curr_x+sw] = lane_curves
	curr_x += sw + pad
	curr_x += 2*pad

	font = cv2.FONT_HERSHEY_SIMPLEX
	text_color = (255,255,255)
	cv2.putText(img, 'Left Curvature: {:.02f}m'.format(l_curvature), (curr_x, 2*pad), font, 0.7, text_color, 1, cv2.LINE_AA)
	cv2.putText(img, 'Right Curvature: {:.02f}m'.format(r_curvature), (curr_x, 4*pad), font, 0.7, text_color, 1, cv2.LINE_AA)
	cv2.putText(img, 'Offset: {:.02f}m'.format(offset), (curr_x, 6*pad), font, 0.7, text_color, 1, cv2.LINE_AA)

	return img

def pipeline(img, is_running_on_video=True):
	global left_curve, right_curve, frame_idx
	img_shape = img.shape
	# Undistort
	undistorted = undistort(img, mtx, dist)
	# Binarize
	binary_result, color_binary_result = gen_binary_image(undistorted)
	# Bird's Eye View
	M, M_inv, warped = warp_top_view(binary_result)
	# Find Lanes
	if frame_idx == 0:
		left_curve, right_curve, warped_lane_curves = find_lane_curves(warped, left_curve, right_curve)
	else:
		left_curve, right_curve, warped_lane_curves = find_lane_curves_seed(warped, left_curve, right_curve)
	# Draw Lanes on Road
	img_lane_curves = draw_lane_curves_on_road(undistorted, M_inv, left_curve, right_curve)
	# Find curvatures and offset
	l_curvature = left_curve.curvature_metres()
	r_curvature = right_curve.curvature_metres()
	offset_from_center = get_offset_from_center(img_shape, left_curve, right_curve)
	# Combine all results
	result = stitch_all_results(img_lane_curves, binary_result, warped, warped_lane_curves, l_curvature, r_curvature, offset_from_center)
	
	if is_running_on_video:
		frame_idx += 1

	return result

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Advanced Lane Finding')
	parser.add_argument('--mode', default='video', type=str,
	                help='img/video')

	args = parser.parse_args()

	calib_dir = 'camera_cal'
	test_img_dir = 'test_images'
	output_dir = 'output_images'
	left_curve = Curve()
	right_curve = Curve()
	frame_idx = 0
	
	mtx, dist = calibrate('camera_cal')

	if args.mode == 'img':
		for f in os.listdir(test_img_dir):
			print ("Processing - ", f)
			file_name = os.path.splitext(f)[0]
			img = mpimg.imread(os.path.join(test_img_dir, f))
			result = pipeline(img, is_running_on_video=False)
			cv2.imwrite(os.path.join(output_dir, file_name + "_result.jpg"), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

	if args.mode == 'video':
		clip = VideoFileClip("project_video.mp4")
		white_clip = clip.fl_image(pipeline)
		white_clip.write_videofile(os.path.join(output_dir, "project_video.mp4") , audio=False)
