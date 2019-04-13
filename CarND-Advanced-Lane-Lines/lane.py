import os
import cv2  
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from utils import write_img_pair
from calib import calibrate, undistort
from binary import gen_binary_image
from perspective import warp_top_view
from curve import Curve

ym_per_pix = 30/(720) # meters per pixel in y dimension
xm_per_pix = 3.7*2/(1280) # meters per pixel in x dimension (while warping the crop extends to half a lane on both sides)

def find_lane_curves(img, left_curve, right_curve, frame_idx=0):
    out_img = np.dstack((img, img, img))
    img_shape = img.shape
    # Find histogram of the lower half of the image, and find the peaks of both halves along he x axis.
    # These will server as starting points for the lane lines  
    histogram = np.sum(img[img_shape[0]//2:,:], axis=0)
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    nwindows = 9 # Number of windows along the height
    margin = 100 # Width of the windows
    minpix = 50 # Minumum pixels found to recenter window

    window_height = np.int(img_shape[0]//nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img_shape[0] - (window+1)*window_height
        win_y_high = img_shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)
        & (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]
        good_right_inds = ((nonzerox >= win_xright_low) & (nonzerox < win_xright_high)
        & (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If found good indices exceeds minpix recenter next window as the mean coordinates at the indices
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Check if lane lines are detected
    if (leftx.shape[0] > 0) and (rightx.shape[0] > 0) and (lefty.shape[0] > 0) and (righty.shape[0] > 0):
        # Fit a second order polynomial to the pixel positions and update the line
        left_fit = np.polyfit(lefty, leftx, 2)
        left_fit_meter = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        right_fit_meter = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        if frame_idx == 0:
            left_curve.update_hard(left_fit, left_fit_meter)
            right_curve.update_hard(right_fit, right_fit_meter)
        else:
            left_curve.update_soft(left_fit, left_fit_meter)
            right_curve.update_soft(right_fit, right_fit_meter)

    # Get the updated lines
    left_fit = left_curve.curve_pixel
    right_fit = right_curve.curve_pixel

    #### VISUALISATION ####
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Label lane pixels ##
    out_img[lefty, leftx] = [0, 255, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Draw lane center
    left_pts = np.vstack((left_fitx, ploty)).astype(np.int32)
    left_pts = left_pts.T
    right_pts = np.vstack((right_fitx, ploty)).astype(np.int32)
    right_pts = right_pts.T
    cv2.polylines(out_img, [left_pts],  False,  (255, 255, 0),  2)
    cv2.polylines(out_img, [right_pts],  False,  (255, 255, 0),  2)

    return left_curve, right_curve, out_img

def find_lane_curves_seed(img, left_curve, right_curve):
    out_img = np.dstack((img, img, img))
    window_img = np.zeros_like(out_img)
    img_shape = img.shape
    left_fit = left_curve.curve_pixel
    right_fit = right_curve.curve_pixel

    # HYPERPARAMETERS
    margin = 100

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_fit_nonzerox = left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2]
    right_fit_nonzerox = right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2]
    left_lane_inds = ((nonzerox > (left_fit_nonzerox - margin)) & 
        (nonzerox < (left_fit_nonzerox + margin)))
    right_lane_inds = ((nonzerox > (right_fit_nonzerox - margin)) & 
        (nonzerox < (right_fit_nonzerox + margin)))

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Check if lane lines are detected
    if (leftx.shape[0] > 0) and (rightx.shape[0] > 0) and (lefty.shape[0] > 0) and (righty.shape[0] > 0):
        # Fit a second order polynomial to the pixel positions and update the line
        left_fit = np.polyfit(lefty, leftx, 2)
        left_fit_meter = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        right_fit_meter = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        # Update line
        left_curve.update_soft(left_fit, left_fit_meter)
        right_curve.update_soft(right_fit, right_fit_meter)        
            
    # Get the updated lines
    left_fit = left_curve.curve_pixel
    right_fit = right_curve.curve_pixel

    #### VISUALISATION ####
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Label lane pixels ##
    out_img[lefty, leftx] = [0, 255, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_window_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_window_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_window_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_window_pts]), (0,255, 0))
    out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # Draw lane center
    left_pts = np.vstack((left_fitx, ploty)).astype(np.int32)
    left_pts = left_pts.T
    right_pts = np.vstack((right_fitx, ploty)).astype(np.int32)
    right_pts = right_pts.T
    cv2.polylines(out_img, [left_pts],  False,  (255, 255, 0),  2)
    cv2.polylines(out_img, [right_pts],  False,  (255, 255, 0),  2)

    return left_curve, right_curve, out_img

def draw_lane_curves_on_road(img, M_inv, left_curve, right_curve):
    img_shape = img.shape
    left_fit = left_curve.curve_pixel
    right_fit = right_curve.curve_pixel

    ploty = np.linspace(0, img_shape[0]-1, img_shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    # Draw lane lines
    road_lines = np.zeros_like(img, dtype=np.uint8)
    # Build format suitable for cv2.fillPoly
    left_pts = np.vstack((left_fitx, ploty)).astype(np.int32)
    left_pts = left_pts.T
    right_pts = np.vstack((right_fitx, ploty)).astype(np.int32)
    right_pts = right_pts.T
    # Draw polylines
    cv2.polylines(road_lines, [left_pts],  False,  (0, 255, 0),  40)
    cv2.polylines(road_lines, [right_pts],  False,  (0, 0, 255),  40)

    # Draw lane window
    lane_left_window = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    lane_right_window = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    lane_window = np.hstack((lane_left_window, lane_right_window))
    cv2.fillPoly(road_lines, np.int_([lane_window]), (255, 0, 0))

    road_lines = cv2.warpPerspective(road_lines, M_inv, (img_shape[1], img_shape[0]))

    result = cv2.addWeighted(img, 1.0, road_lines, 1.0, 0)

    return result

def get_offset_from_center(img_shape, left_curve, right_curve):
    y_eval = img_shape[0]
    left_fit = left_curve.curve_pixel
    right_fit = right_curve.curve_pixel
    left_fitx = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    right_fitx = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    lane_width = right_fitx - left_fitx
    offset_pixel = abs(img_shape[1]/2 - (left_fitx + lane_width/2))
    offset_metres = xm_per_pix*offset_pixel
    return offset_metres

if __name__ == '__main__':

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
    left_curve = Curve()
    right_curve = Curve()

    left_curve, right_curve, lane_curves = find_lane_curves(warped, left_curve, right_curve)
    write_img_pair(warped, lane_curves, "Warped Image", "Image showing Lane Lines", 
        os.path.join(output_dir, file_name + "_warped_lane_result.jpg"))

    left_curve, right_curve, lane_curves = find_lane_curves_seed(warped, left_curve, right_curve)
    write_img_pair(warped, lane_curves, "Warped Image", "Image showing Lane Lines", 
        os.path.join(output_dir, file_name + "_warped_lane_seed_result.jpg"))