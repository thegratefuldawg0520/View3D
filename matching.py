import cv2
import keypoints as kp

class matches(object):

	def __init__(self,lImage,rImage,params):
		
		self.lImage = lImage
		self.rImage = rImage
		self.params = params
		self.matches = self._getMatches()
		
	def _getMatches(self):
	
		if self.params['kp'] == 'orb' or self.params['kp'] == 'brisk':
			
			bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
			binMatch = bf.match(self.lImage.descriptors,self.rImage.descriptors)
			
			good = []
			
			for match in binMatch:
			
				if match.distance < 40:
			
					good.append(match)
					
			return good
			
		elif self.params['kp'] == 'sift' or self.params['kp'] == 'surf':
		
			index_params = dict(algorithm = 0, trees = 5)
			search_params = dict(checks=50)
			flann = cv2.FlannBasedMatcher(index_params, search_params)
			flannMatch = flann.knnMatch(self.lImage.descriptors, self.rImage.descriptors, k=2)
			
			good = []
			
			for m,n in flannMatch:
			
				if m.distance < 0.75*n.distance:
					
					good.append([m])
			
			return good

	def drawMatches(self):
	
		if self.matches == None:
			
			return False
			
		else:
			
			if self.params['kp'] == 'orb' or self.params['kp'] == 'brisk':
				
				print self.matches
				img = cv2.drawMatches(self.lImage.image,self.lImage.keypoints,self.rImage.image,self.rImage.keypoints,self.matches, None,flags=2)
				for i in self.rImage.keypoints:
				
					print i.pt
				
			elif self.params['kp'] == 'sift' or self.params['kp'] == 'surf':
		
				img = cv2.drawMatchesKnn(self.lImage.image,self.lImage.keypoints,self.rImage.image,self.rImage.keypoints,self.matches[:10], None,flags=2)
				
			cv2.imshow("Matches", img)
			cv2.waitKey()
			cv2.destroyAllWindows()		

if __name__ == '__main__':

	param = {'kp':'surf','scale':0.15}
	
	imgr = kp.keypoints(cv2.imread('/home/doopy/Documents/View3D/View3D_0_1/0214.JPG'),param)
	imgl = kp.keypoints(cv2.imread('/home/doopy/Documents/View3D/View3D_0_1/0215.JPG'),param)
	
	rlMatch = matches(imgl,imgr,param)
	
	rlMatch.drawMatches()
	
			

