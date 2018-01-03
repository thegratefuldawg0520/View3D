import cv2

class image(object):

	def __init__(self, img, params):
		
		self.img = cv2.resize(img, None, fx=params['scale'], fy=params['scale'], interpolation=cv2.INTER_AREA)
		self.params = params
		self.detector = self._getDetector(self.params)
		self.keypoints = self._getKeypoints()
		self.descriptors = self._getDescriptors()
		self.path = img
		
	def _getDetector(self,params):
		
		if params['kp'] == 'sift':
			
			return cv2.xfeatures2d.SIFT_create()
			
		elif params['kp'] == 'surf':
		
			return cv2.xfeatures2d.SURF_create()

		elif params['kp'] == 'orb':
		
			return cv2.ORB_create()
		
		elif params['kp'] == 'brisk':
		
			return cv2.BRISK_create()

	def _getKeypoints(self):
	
		return self.detector.detect(self.img,None)
		
	def _getDescriptors(self):

		return self.detector.compute(self.img,self.keypoints)[1]
