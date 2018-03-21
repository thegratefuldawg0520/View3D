import cv2
import numpy as np

class matches(object):

	def __init__(self,img1,img2,K,params):
		
		self.img1 = img1
		self.img2 = img2
		self.params = params
		self.matches = self._getMatches()
		self.matchPoints = self._sortMatchPoints()
		self.K = K
		self.P = np.hstack((np.eye(3), np.zeros((3, 1))))
		
	def _getMatches(self):
	
		if self.params['kp'] == 'orb' or self.params['kp'] == 'brisk' or self.params['kp'] == 'freak' or self.params['kp'] == 'lucid':
			
			bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
			binMatch = bf.match(self.img1.descriptors,self.img2.descriptors)
			
			good = []
			
			for match in binMatch:
			
				if match.distance < 40:
			
					good.append(match)
					
			return good
			
		else:
		
			index_params = dict(algorithm = 0, trees = 5)
			search_params = dict(checks=50)
			flann = cv2.FlannBasedMatcher(index_params, search_params)
			flannMatch = flann.knnMatch(self.img1.descriptors, self.img2.descriptors, k=2)
			
			good = []
			
			for m,n in flannMatch:
			
				if m.distance < 0.7*n.distance:
					
					good.append([m])
			
			return good

	def _sortMatchPoints(self):
		
		if self.params['kp'] == 'orb' or self.params['kp'] == 'brisk' or self.params['kp'] == 'freak' or self.params['kp'] == 'lucid':
			
			img1_pts = np.float32([ self.img1.keypoints[match.queryIdx].pt for match in self.matches]).reshape(-1,1,2)
			img2_pts = np.float32([ self.img2.keypoints[match.trainIdx].pt for match in self.matches]).reshape(-1,1,2)

		else:
		
			img1_pts = np.float32([ self.img1.keypoints[match[0].queryIdx].pt for match in self.matches]).reshape(-1,1,2)
			img2_pts = np.float32([ self.img2.keypoints[match[0].trainIdx].pt for match in self.matches]).reshape(-1,1,2)

		return {'img1':img1_pts,'img2':img2_pts}
		
	def drawMatches(self):
	
		if self.matches == None:
			
			return False
			
		else:
			
			if self.params['kp'] == 'orb' or self.params['kp'] == 'brisk' or self.params['kp'] == 'lucid' or self.params['kp'] == 'freak':
				
				img = cv2.drawMatches(self.img1.img,self.img1.keypoints,self.img2.img,self.img2.keypoints,self.matches, None,flags=2)
		
			elif self.params['kp'] == 'sift' or self.params['kp'] == 'surf' or self.params['kp'] == 'kaze' or self.params['kp'] == 'daisy':
		
				img = cv2.drawMatchesKnn(self.img1.img,self.img1.keypoints,self.img2.img,self.img2.keypoints,self.matches, None,flags=2)
				
			cv2.imshow("Matches", img)
			cv2.waitKey()
			cv2.destroyAllWindows()		

	def eulerAngles(self):
		
		P = self.P


		p = np.arcsin(P[0,2])
		o = np.arctan2(-P[1,2],P[2,2])
		k = np.arctan2(-P[0,1],P[0,0])
			
		return np.array([o,p,k])
		
	def rMatrix(self,o,p,k):
		
		coso = np.cos(o)
		sino = np.sin(o)
		cosp = np.cos(p)
		sinp = np.sin(o)
		cosk = np.cos(k)
		sink = np.sin(o)
		
		r11 = cosp*cosk
		r12 = -cosp*sink
		r13 = sinp
		r21 = coso*sink + sino*sinp*cosk
		r22 = coso*cosk - sino*sinp*sink
		r23 = -sino*cosp
		r31 = sino*sink - coso*sinp*cosk
		r32 = sino*cosk + coso*sinp*sink
		r33 = coso*cosp
		
		return np.array([[r11,r12,r13],[r21,r22,r23],[r31,r32,r33]])
