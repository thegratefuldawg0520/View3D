import cv2
import matching as mt
import imagek as im
import numpy as np

class transformation(object):
	
	def __init__(self, img1, img2, params, matches=None):
	
		self.img1 = self._getKeypoints(img1,params)
		self.img2 = self._getKeypoints(img2,params)
		self.matches = self._getMatches(matches,self.img1,self.img2,params)

	def _getKeypoints(self,img,params):
	
		if isinstance(img, im.image):
		
			return img
			
		return im.image(img,params)
					
	def _getMatches(self,matches,img1,img2,params):
	
		if matches != None:
		
			return matches

		return mt.matches(img1,img2,params)
		
	def imageNames(self):
		
		return "image 1: " + self.img1.path + "\nimage 2: " + self.img2.path
			
class homography(transformation):

	def __init__(self, img1, img2, params, matches=None):
	
		super(homography,self).__init__(img1, img2, params, matches)
		self.homography = self._computeHomography(params)
		
	def _computeHomography(self,params):
	
		img1_pts = np.float32([ self.img1.keypoints[match.queryIdx].pt for match in self.matches.matches ]).reshape(-1,1,2)
		img2_pts = np.float32([ self.img2.keypoints[match.trainIdx].pt for match in self.matches.matches ]).reshape(-1,1,2)
		
		return cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)
