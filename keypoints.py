import cv2

class keypoints(object):

	def __init__(self, image, params):
	
		self.image = cv2.resize(image, None, fx=params['scale'], fy=params['scale'], interpolation=cv2.INTER_AREA)
		self.params = params
		self.detector = self._getDetector(self.params)
		self.keypoints = self._getKeypoints()
		self.descriptors = self._getDescriptors()
		
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
	
		return self.detector.detect(self.image,None)
		
	def _getDescriptors(self):
	
		return self.detector.compute(self.image,self.keypoints)[1]

if __name__ == '__main__':

	img = cv2.imread('/home/doopy/Documents/View3D/View3D_0_1/boxes.jpg',0)
	kd = keypoints(img,{'kp':'brisk'})
	print type(kd.detector)
	
