import cv2
import matching as mt
import image as im
import imageUtility as ut
import time
import numpy as np
class transformation(object):
	
	def __init__(self, img1, img2, params, matches=None):
	
		self.img1 = self._getKeypoints(img1,params)
		self.img2 = self._getKeypoints(img2,params)
		self.matches = self._getMatches(matches,self.img1,self.img2,params)
		self.bitmask = [0]

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
		
	def matchCount(self):
		
		return len(self.matches.matches)
		
	def inlierCount(self):
		
		return len(self.bitmask[self.bitmask == 1])
		
	def outlierCount(self):
		
		return len(self.bitmask[self.bitmask == 0])
			
class homography(transformation):

	def __init__(self, img1, img2, params, matches=None):
	
		super(homography,self).__init__(img1, img2, params, matches)
		self.homography,self.bitmask = self._computeHomography(params)
		
	def _computeHomography(self,params):
	
		img1pts = self.matches.matchPoints['img1']
		img2pts = self.matches.matchPoints['img2']
		
		return cv2.findHomography(img1pts, img2pts, cv2.RANSAC, 5.0)

class fundamental(transformation):
	
	def __init__(self, img1, img2, params, matches=None):
	
		super(fundamental,self).__init__(img1, img2, params, matches)
		self.fundamental,self.bitmask = self._computeFundamental(params)
		
	def _computeFundamental(self,params):
	
		img1pts = self.matches.matchPoints['img1']
		img2pts = self.matches.matchPoints['img2']
		
		return cv2.findFundamentalMat(img1pts, img2pts, cv2.RANSAC)
	
	def getEssential(self):
		
		return self.img1.K.T.dot(self.fundamental).dot(self.img2.K)
	
	def getCameraMatrices(self):
		
		E = self.getEssential()
		
		U, S, Vt = np.linalg.svd(E)
		
		W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
		
		kp1 = []
		kp2 = []
		
		for i in range(len(self.img1.keypoints)):
			
			kp1.append([self.img1.keypoints[i].pt[0],self.img1.keypoints[i].pt[1],1.0])
			kp2.append([self.img2.keypoints[i].pt[0],self.img2.keypoints[i].pt[1],1.0])
		
		R = U.dot(W).dot(Vt)
		T = U[:,2]
		
		if not self._in_front(kp1,kp2,R,T):
			T = -U[:,2]
		
		if not self._in_front(kp1,kp2,R,T):
			R = U.dot(W.T).dot(Vt)
			T = U[:,2]
			
		if not self._in_front(kp1,kp2,R,T):
			T = -U[:,2]
		
		Rt1 = np.hstack((np.eye(3), np.zeros((3, 1))))
		Rt2 = np.hstack((R, T.reshape(3, 1)))

		return Rt2
		
	def _in_front(self,kp1,kp2,R,T):
		
		rot = R
		trans = T
		for first, second in zip(kp1, kp2):

			first_z = np.dot(rot[0, :] - second[0]*rot[2, :], trans) / np.dot(rot[0, :] - second[0]*rot[2, :], second)
			first_3d_point = np.array([first[0] * first_z, second[0] * first_z, first_z])
			second_3d_point = np.dot(rot.T, first_3d_point) - np.dot(rot.T, trans)
			
		if first_3d_point[2] < 0 or second_3d_point[2] < 0:
			
			return False
		
		return True
			
	def decompose(self):
		
		return cv2.RQDecomp3x3(self.fundamental)

#cv2.decomposeProjectionMatrix

if __name__=="__main__":
	
	params = {'scale':0.15,'kp':'sift'}
	
	img1 = im.image('/home/doopy/Documents/View3D/View3D_0_1/Glacier/img/EP-00-00019_0044_0002.JPG',params)
	
	img2 = im.image('/home/doopy/Documents/View3D/View3D_0_1/Glacier/img/EP-00-00019_0044_0002.JPG',params)
	
	F = fundamental(img1,img2,params)

	print F.getCameraMatrices()
	
