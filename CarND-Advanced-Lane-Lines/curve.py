import os
import numpy as np

class Curve:

	def __init__(self):

		self.curve_pixel = [0,0,0]
		self.curve_meter = [0,0,0]

	def update_hard(self, curve_pixel, curve_meter):
		self.curve_pixel = curve_pixel
		self.curve_meter = curve_meter

	def update_soft(self, curve_pixel, curve_meter):
		# Moving average
		for i, coeff in enumerate(curve_pixel):
			self.curve_pixel[i] = self.curve_pixel[i]*0.8 + coeff*0.2
		for i, coeff in enumerate(curve_meter):
			self.curve_meter[i] = self.curve_meter[i]*0.8 + coeff*0.2

	def curvature_pixel(self):
		y_eval = 0
		curverad = ((1+(2*self.curve_pixel[0]*y_eval+self.curve_pixel[1])**2)**1.5)/np.absolute(2*self.curve_pixel[0])
		return curverad

	def curvature_metres(self):
		y_eval = 0
		curverad = ((1+(2*self.curve_meter[0]*y_eval+self.curve_meter[1])**2)**1.5)/np.absolute(2*self.curve_meter[0])
		return curverad
