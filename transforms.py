import cv2
import matching as mt
import image as im
import imageUtility as ut
import time
import numpy as np
import matplotlib.pyplot as plt

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
	
		return cv2.findFundamentalMat(self.matches.matchPoints['img1'][:,0], self.matches.matchPoints['img2'][:,0], cv2.RANSAC)
	
	def getEssential(self):
		
		temp = np.dot(self.img1.K.T,np.dot(self.fundamental,self.img2.K))
		
		U, S, Vt = np.linalg.svd(temp)
		
		#Why?!?
		if np.linalg.det(np.dot(U,Vt)) < 0:
			
			Vt = -Vt
			
		E = np.dot(U, np.dot(np.diag([1,1,0]), Vt))
		
		return E,U,np.diag([1,1,0]),Vt
		
	def getCameraMatrices(self,i):
		
		E,U,S,Vt = self.getEssential()
		print E
		print np.dot(self.img1.K.T,np.dot(self.fundamental,self.img2.K))
		print cv2.findEssentialMat(self.matches.matchPoints['img1'][:,0], self.matches.matchPoints['img2'][:,0], self.img1.K, cv2.RANSAC)[0]
		input()
		W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
		
		R = np.dot(U,np.dot(W,Vt))
		T = U[:,2]
		
		P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
		P2 = np.dot(self.img2.K,np.hstack((R, T.reshape(3, 1))))
		print P2
		
		if not self._in_front_of_camera(P1,P2,self.matches.matchPoints['img1'][:,0,:],self.matches.matchPoints['img2'][:,0,:],str(i) + 'a'):
		 
			T = -T
			P2 = np.dot(self.img2.K,np.hstack((R, T.reshape(3, 1))))
			print P2
		 
		if not self._in_front_of_camera(P1,P2,self.matches.matchPoints['img1'][:,0,:],self.matches.matchPoints['img2'][:,0,:],str(i) + 'b'):
		 
			R = U.dot(W.T).dot(Vt)
			P2 = np.dot(self.img2.K,np.hstack((R, T.reshape(3, 1))))
			print P2
		if not self._in_front_of_camera(P1,P2,self.matches.matchPoints['img1'][:,0,:],self.matches.matchPoints['img2'][:,0,:],str(i) + 'c'):
			 
			T = -T
			P2 = np.dot(self.img2.K,np.hstack((R, T.reshape(3, 1))))
			self._in_front_of_camera(P1,P2,self.matches.matchPoints['img1'][:,0,:],self.matches.matchPoints['img2'][:,0,:],str(i) + 'd')
			print P2
		print '*****************************'
		

	def _in_front_of_camera(self,P1,P2,kp1,kp2,title):
		
		pts = cv2.triangulatePoints(P1,P2,kp1.T,kp2.T)
		
		#pts = np.array([pts[0]/pts[3],pts[1]/pts[3],pts[2]/pts[3]])
		T = pts[3]
		print T
		M = P1[:,0:3]
		print M
		p4 = P1[:,3]
		print p4
		m3 = M[2,:]
		print m3
		
		x = np.dot(P1,pts)
		print x
		w = x[2]
		print w
		print pts
		pts = pts/w
		print pts
		print kp1
		input()
		pts[3] = np.sign(np.linalg.det(M)*w)/(T*np.linalg.norm(m3))
		
		ut.createPCFile(pts.T, '/home/doopy/Documents/View3D/View3D_0_1/pc' + title + '.txt')
		
	def _check_epipolar(E,kp1,kp2):
		
		for i,j in zip(kp2,kp1):
			
			#Epipolar Condition
			print i.dot(E).dot(j)
 
if __name__=="__main__":
	
	nOctaveLayers = 3
	dOctaveLayers = 1
	
	contrastThreshold = 0.04
	dContrastThreshold = 0.01
	
	edgeThreshold = 10
	dEdgeThreshold = 1
	
	sigma = 1.6
	dSigma = 0.1
	
	params = {'scale':1.0,
			  'kp':'sift',
			  'nOctaveLayers':nOctaveLayers,
			  'contrastThreshold':contrastThreshold,
			  'edgeThreshold':edgeThreshold,
			  'sigma':sigma
			 }
	K = np.array([[1520.4, 0., 302.32], [0, 1525.9, 246.87], [0, 0, 1]])
	img1 = im.image('/home/doopy/Documents/View3D/View3D_0_1/templeRing/templeR0001.png',params)
	img1.setK(K)
    
	for i in range(1,5):
 
		img2 = im.image('/home/doopy/Documents/View3D/View3D_0_1/templeRing/templeR000'  + str(i + 1) + '.png',params)
		img2.setK(K)

		match = mt.matches(img1,img2,params)
		match.drawMatches()
		E = cv2.findEssentialMat(match.matchPoints['img1'][:,0], match.matchPoints['img2'][:,0], K, cv2.RANSAC)[0]
		print match.matchPoints['img1'][:,0]
		correctedMatches = cv2.correctMatches(E,match.matchPoints['img1'][:,0],match.matchPoints['img2'][:,0])
		pose = cv2.recoverPose(E, match.matchPoints['img1'][:,0], match.matchPoints['img2'][:,0],K,True)
		P1 = np.dot(K,np.hstack((np.eye(3),np.zeros((3, 1)))))
		P2 = np.dot(K,np.hstack((pose[1],pose[2])))
		print P1
		pts = cv2.triangulatePoints(P1,P2,match.matchPoints['img1'][:,0].T,match.matchPoints['img2'][:,0].T)
		pts = np.array([pts[0]/pts[3],pts[1]/pts[3],pts[2]/pts[3]])
		ut.createPCFile(pts.T, '/home/doopy/Documents/View3D/View3D_0_1/pc' + str(i) + '.txt')

		img1 = img2
		#F = fundamental(img1,img2,params)
		
		#F.getCameraMatrices(i)
