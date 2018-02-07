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
		
		R = U.dot(W).dot(Vt)
		
		T = U[:,2]
		
		if round(np.linalg.det(R)) == -1.0:
			
			R = -R
			
		if round(np.linalg.det(R)) != 1.0 or round(T.dot(T)) != 1.0:
			
			print 'not a valid decomposition'
			
			return False

		P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
		P2 = np.hstack((R, T.reshape(3, 1)))
		
		if not self._in_front_of_camera(P1,P2,self.matches.matchPoints['img1'][:,0,:],self.matches.matchPoints['img2'][:,0,:],'a'):
			
			T = -T
			P2 = np.hstack((R, T.reshape(3, 1)))
		
		if not self._in_front_of_camera(P1,P2,self.matches.matchPoints['img1'][:,0,:],self.matches.matchPoints['img2'][:,0,:],'b'):
			
			R = U.dot(W.T).dot(Vt)
			P2 = np.hstack((R, T.reshape(3, 1)))
		
		if not self._in_front_of_camera(P1,P2,self.matches.matchPoints['img1'][:,0,:],self.matches.matchPoints['img2'][:,0,:],'c'):
			
			T = -T
			P2 = np.hstack((R, T.reshape(3, 1)))
			self._in_front_of_camera(P1,P2,self.matches.matchPoints['img1'][:,0,:],self.matches.matchPoints['img2'][:,0,:],'d')
			
		print '*****************************'
		

	def _in_front_of_camera(self,P1,P2,kp1,kp2,title):
		
		pts = cv2.triangulatePoints(P1,P2,kp1.T,kp2.T)
		
		ut.createPCFile(pts.T, '/home/dennis/Documents/View3D/pc' + title + '.txt')
		
	def _check_epipolar(E,kp1,kp2):
		
		for i,j in zip(kp2,kp1):
			
			#Epipolar Condition
			print i.dot(E).dot(j)
			
if __name__=="__main__":
	
	for i in range(0,1):
		
		params = {'scale':0.15,'kp':'sift'}
		
		img1 = im.image('/home/dennis/Documents/View3D/images/DJI0' + str(1765+i) + '.JPG',params)
		
		img2 = im.image('/home/dennis/Documents/View3D/images/DJI0' + str(1765+i+1) + '.JPG',params)
		
		F = fundamental(img1,img2,params)
		
		#H = homography(img1,img2,params)
	
		P_ = F.getCameraMatrices()
		#P = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
	
	
