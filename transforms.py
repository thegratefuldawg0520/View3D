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
	
		
		n1pts = ut.normalize(self.matches.matchPoints['img1'][:,0])
		n2pts = ut.normalize(self.matches.matchPoints['img2'][:,0])
		
		img1pts = np.ones(self.matches.matchPoints['img1'].shape)
		img2pts = np.ones(self.matches.matchPoints['img2'].shape)
		
		img1pts[:,0,:] = n1pts.T
		img2pts[:,0,:] = n2pts.T
		
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
		
		print kp1.shape
		k1 = ut.normalize(kp1)
		k2 = ut.normalize(kp2)
		
		pts = cv2.triangulatePoints(P1,P2,k1,k2)
		
		pts = np.array([pts[0]/pts[3],pts[1]/pts[3],pts[2]/pts[3]])
		
		ut.createPCFile(pts.T, '/home/doopy/Documents/View3D/View3D_0_1/pc' + title + '.txt')
		
	def _check_epipolar(E,kp1,kp2):
		
		for i,j in zip(kp2,kp1):
			
			#Epipolar Condition
			print i.dot(E).dot(j)
			
if __name__=="__main__":
	
	for i in range(0,1):
		
		params = {'scale':0.15,'kp':'sift'}
		
		img1 = im.image('/home/doopy/Documents/View3D/View3D_0_1/Glacier/img/EP-00-00019_0044_000'  + str(2*i + 2) + '.JPG',params)
		
		img2 = im.image('/home/doopy/Documents/View3D/View3D_0_1/Glacier/img/EP-00-00019_0044_000'  + str(2*i + 3) + '.JPG',params)
		
		F = fundamental(img1,img2,params)
		#F.matches.drawMatches()
		#H = homography(img1,img2,params)
	
		P_ = F.getCameraMatrices()
		#P = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
	
	
