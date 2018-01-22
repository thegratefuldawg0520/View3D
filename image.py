import cv2

#https://docs.opencv.org/2.4/modules/features2d/doc/feature_detection_and_description.html
#https://docs.opencv.org/3.3.0/d3/df6/namespacecv_1_1xfeatures2d.html
#https://docs.opencv.org/3.3.0/d2/d75/namespacecv.html
#https://docs.opencv.org/3.1.0/d3/df6/namespacecv_1_1xfeatures2d.html
class image(object):

	def __init__(self, img, params):
		
		tempImg = cv2.imread(img)
		self.img = cv2.resize(tempImg, None, fx=params['scale'], fy=params['scale'], interpolation=cv2.INTER_AREA)
		self.params = params
		self.detector = self._getDetector(self.params)
		self.keypoints = self._getKeypoints()
		self.descriptors = self._getDescriptors()
		self.path = img
		
	def _getDetector(self,params):
		
		if params['kp'] == 'sift':
			
			return cv2.xfeatures2d.SIFT_create(nfeatures=0,nOctaveLayers=3,contrastThreshold=0.04,edgeThreshold=10,sigma=1.6)
			
		elif params['kp'] == 'surf':
			
			return cv2.xfeatures2d.SURF_create(nOctaves=4,nOctaveLayers=2)

		elif params['kp'] == 'orb':
		
			return cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0, WTA_K=2, patchSize=31)
		
		elif params['kp'] == 'brisk':
		
			return cv2.BRISK_create(thresh=30, octaves=3, patternScale=1.0)

		elif params['kp'] == 'kaze':
			
			return cv2.KAZE_create(threshold=0.001, nOctaves=4, nOctaveLayers=4)

		elif params['kp'] == 'daisy':
			
			return cv2.DAISY_create(radius=15,q_radius=3,q_theta=8,q_hist=8)

		elif params['kp'] == 'freak':
			
			return cv2.FREAK_create(patternScale=22.0, nOctaves=4)

		elif params['kp'] == 'lucid':
			
			return cv2.LUCID_create(lucid_kernel=1,blur_kernel=1)		
			
	def _getKeypoints(self):

		return self.detector.detect(self.img,None)
		
	def _getDescriptors(self):
		
		return self.detector.compute(self.img,self.keypoints)[1]
