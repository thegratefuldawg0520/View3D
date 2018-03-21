import cv2
import matching as mt
import image as im
import imageUtility as ut
import time
import numpy as np
import matplotlib.pyplot as plt

class transformation(object):
	
	def __init__(self, img1, img2, K, params, matches=None):
	
		self.img1 = self._getKeypoints(img1,params)
		self.img2 = self._getKeypoints(img2,params)
		self.matches = self._getMatches(matches,self.img1,self.img2,K,params)
		self.matches.K = K
		self.bitmask = [0]

	def _getKeypoints(self,img,params):
	
		if isinstance(img, im.image):
		
			return img
			
		return im.image(img,params)
					
	def _getMatches(self,matches,img1,img2,K,params):
	
		if matches != None:
		
			return matches

		return mt.matches(img1,img2,K,params)
		
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
		self.pt3D = None
		
	def _computeFundamental(self,params):
	
		F, mask = cv2.findFundamentalMat(self.matches.matchPoints['img1'][:,0], self.matches.matchPoints['img2'][:,0], cv2.RANSAC)
		
		assert(np.linalg.det(F) < 1.0e6)
		
		return F,mask
	
	def getEssential(self):
		
		temp = np.dot(self.img1.K.T,np.dot(self.fundamental,self.img2.K))
		
		U, S, Vt = np.linalg.svd(temp)
		
		if np.linalg.det(np.dot(U,Vt)) < 0:
			
			Vt = -Vt
			
		E = np.dot(U, np.dot(np.diag([1,1,0]), Vt))
		V = Vt.T
		
		assert(np.linalg.det(U) > 0)
		assert(np.linalg.det(V) > 0)
		assert sum(np.dot(E, np.dot(E.T, E)) - 0.5 * np.trace(np.dot(E, E.T)) * E)[0] < 1.0e-10
		
		return E

	def _depth(self,X,P):
		
		T = X[3]
		M = P[:,0:3]
		p4 = P[:,3]
		m3 = M[2,:]
		
		x = np.dot(P,X)
		w = x[2]
		X = X/w
		
		return (np.sign(np.linalg.det(M)) * w) / (T*np.linalg.norm(m3))
		
	def _computeTriangulation(self,pt1,P1,pt2,P2):
		
		pt1 = pt1/pt1[2]
		x1, y1 = pt1[0], pt1[1]
		
		pt2 = pt2/pt2[2]
		x2, y2 = pt2[0], pt2[1]
		
		a0 = x1*P1[2,:] - P1[0,:]
		a1 = y1*P1[2,:] - P1[0,:]
		a2 = x2*P2[2,:] - P2[0,:]
		a3 = y2*P2[2,:] - P2[0,:]
		
		A = np.vstack((a0,a1,a2,a3))
		U, S, Vt = np.linalg.svd(A)
		V = Vt.T
		
		X3d = V[:,-1]
		
		return X3d/X3d[3]
		
		
	def getCameraMatrices(self):
		
		E = self.getEssential()
		E = cv2.findEssentialMat(self.matches.matchPoints['img1'][:,0], self.matches.matchPoints['img2'][:,0], self.matches.K, cv2.RANSAC)[0]
		
		W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
		
		U,S,Vt = np.linalg.svd(E)
		T = U[:,2]
		
		P1 = np.dot(self.img1.K,np.hstack((np.eye(3), np.zeros((3, 1)))))
		
		Pi = []
		Pi.append(np.dot(self.img2.K,np.hstack((np.dot(U, np.dot(W ,Vt)), T.reshape(3, 1)))))
		Pi.append(np.dot(self.img2.K,np.hstack((np.dot(U, np.dot(W ,Vt)), -T.reshape(3, 1)))))
		Pi.append(np.dot(self.img2.K,np.hstack((np.dot(U, np.dot(W.T ,Vt)), T.reshape(3, 1)))))
		Pi.append(np.dot(self.img2.K,np.hstack((np.dot(U, np.dot(W.T ,Vt)), -T.reshape(3, 1)))))
		
		pts1 = ut.toHomogeneous(self.matches.matchPoints['img1'][:,0])
		pts2 = ut.toHomogeneous(self.matches.matchPoints['img2'][:,0])
		
		P2 = None
		
		for P in Pi:
			
			Q = self._computeTriangulation(pts1[0],P1,pts2[0],P)
			
			if self._depth(Q,P1) > 0 and self._depth(Q,P) > 0:
				
				P2 = P
				break
				
		assert(P2 is not None)
		
		self.matches.P = np.dot(np.linalg.inv(self.img2.K),P2)
		
	def triangulate(self):
		
		pts1 = ut.toHomogeneous(self.matches.matchPoints['img1'][:,0])
		pts2 = ut.toHomogeneous(self.matches.matchPoints['img2'][:,0])
		
		pt3D = []
		
		for i in range(len(pts1)):
			
			pt3D.append(self._computeTriangulation(pts1[i],np.dot(self.img1.K,self.img1.P),pts2[i],np.dot(self.img2.K,self.img2.P)))
		
		self.pt3D = pt3D
		
if __name__=="__main__":
	
	nOctaveLayers = 3
	contrastThreshold = 0.04
	edgeThreshold = 10
	sigma = 1.6
	
	params = {'scale':1.0,
			  'kp':'sift',
			  'nOctaveLayers':nOctaveLayers,
			  'contrastThreshold':contrastThreshold,
			  'edgeThreshold':edgeThreshold,
			  'sigma':sigma
			 }
			 
	K = np.array([[1520.4, 0., 302.32], [0, 1525.9, 246.87], [0, 0, 1]])
	img1 = im.image('/home/doopy/Documents/View3D/View3D_0_1/templeSparseRing/templeSR0001.png',K,params)
	img1.setK(K)
	
	Pglobal = np.vstack((np.hstack((np.eye(3), np.zeros((3, 1)))),np.array([0,0,0,1])))
	
	refData = ut.loadCamerasTemple('/home/doopy/Documents/View3D/View3D_0_1/templeSparseRing/templeSR_par.txt')
	PRefOrigin = np.vstack((np.hstack((refData['templeSR0001.png']['R'],refData['templeSR0001.png']['t'].reshape(3,1))),np.array([0,0,0,1])))
	
	
	for i in range(1,4):
		
		img2 = im.image('/home/doopy/Documents/View3D/View3D_0_1/templeSparseRing/templeSR000' + str(1+i) + '.png',K,params)
		img2.setK(K)
		
		F = fundamental(img1, img2, K, params)
		
		F.getCameraMatrices()
		
		P1 = np.vstack((img1.P,np.array([0,0,0,1])))
		P = np.vstack((F.matches.P,np.array([0,0,0,1])))
		
		Pglobal = np.dot(P,Pglobal)
		
		img2.P = Pglobal[:3,:]
		
		#print np.vstack((,np.array([0,0,0,1])))
		PRef = np.vstack((np.hstack((refData['templeSR000' + str(1+i) + '.png']['R'],refData['templeSR000' + str(1+i) + '.png']['t'].reshape(3,1))),np.array([0,0,0,1])))
		print np.dot(np.linalg.inv(PRefOrigin),PRef)
		print refData['templeSR0002.png']['t'] - refData['templeSR0001.png']['t']
		
		input()
		eA = img1.eulerAngles()
		print 'Left Image'
		print eA
		#print F.matches.rMatrix(eA[0],eA[1],eA[2])
		
		print 'Rotation'
		eA = F.matches.eulerAngles()
		print eA
		
		print 'Right Image'
		eA = img2.eulerAngles()
		print eA
		#print img2.rMatrix(eA[0],eA[1],eA[2])
		input()
		
		
		F.triangulate()
		
		pts = F.pt3D
		
		ut.createPCFile(pts,'outfile' + str(i) + '.txt')
		
		img1 = img2
