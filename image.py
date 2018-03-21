import cv2
import numpy as np

#https://docs.opencv.org/2.4/modules/features2d/doc/feature_detection_and_description.html
#https://docs.opencv.org/3.3.0/d3/df6/namespacecv_1_1xfeatures2d.html
#https://docs.opencv.org/3.3.0/d2/d75/namespacecv.html
#https://docs.opencv.org/3.1.0/d3/df6/namespacecv_1_1xfeatures2d.html
#https://docs.opencv.org/3.3.1/d1/db4/group__xfeatures2d.html

class image(object):

	#TODO: Implement methods to recompute a new detector/descriptor without reloading the image and/or loading multiple copies of the same imagcd e
	def __init__(self, img, K, params):
		
		tempImg = cv2.imread(img)
		self.img = cv2.resize(tempImg, None, fx=params['scale'], fy=params['scale'], interpolation=cv2.INTER_AREA)
		self.params = params
		self.detector,self.desc = self._getDetector(self.params)
		self.keypoints = self._getKeypoints()
		self.descriptors = self._getDescriptors()
		self.path = img
		self.K = K
		self.P = np.hstack((np.eye(3), np.zeros((3, 1))))
		
	def _getDetector(self,params):
		
		if params['kp'] == 'sift':
			
			det_desc = cv2.xfeatures2d.SIFT_create(nfeatures=10000,nOctaveLayers=params['nOctaveLayers'],contrastThreshold=params['contrastThreshold'],edgeThreshold=params['edgeThreshold'],sigma=params['sigma'])
			return det_desc, det_desc
			
		elif params['kp'] == 'surf':
			
			det_desc = cv2.xfeatures2d.SURF_create(nOctaves=params['nOctaves'],nOctaveLayers=params['nOctaveLayers'])
			return det_desc, det_desc

		elif params['kp'] == 'orb':
		
			det_desc = cv2.ORB_create(nfeatures=10000, scaleFactor=params['scaleFactor'], nlevels=params['nlevels'], edgeThreshold=params['edgeThreshold'], firstLevel=params['firstLevel'], WTA_K=params['WTA_K'], patchSize=params['patchSize'])
			return det_desc, det_desc
		
		elif params['kp'] == 'brisk':
			
			det_desc = cv2.BRISK_create(thresh=params['thresh'], octaves=params['octaves'], patternScale=params['patternScale'])
			return det_desc, det_desc

		elif params['kp'] == 'kaze':
			
			det_desc = cv2.KAZE_create(threshold=param['threshold'], nOctaves=param['nOctaves'], nOctaveLayers=param['nOctaveLayers'])
			return det_desc, det_desc

		elif params['kp'] == 'daisy':
			
			return cv2.xfeatures2d.SIFT_create(nfeatures=0,nOctaveLayers=3,contrastThreshold=0.04,edgeThreshold=10,sigma=1.6),cv2.xfeatures2d.DAISY_create(radius=15,q_radius=3,q_theta=8,q_hist=8)

		elif params['kp'] == 'freak':
			
			return cv2.xfeatures2d.SIFT_create(nfeatures=0,nOctaveLayers=3,contrastThreshold=0.04,edgeThreshold=10,sigma=1.6),cv2.xfeatures2d.FREAK_create(patternScale=22.0, nOctaves=4)

		elif params['kp'] == 'lucid':
			
			return cv2.xfeatures2d.SIFT_create(nfeatures=0,nOctaveLayers=3,contrastThreshold=0.04,edgeThreshold=10,sigma=1.6),cv2.xfeatures2d.LUCID_create(lucid_kernel=1,blur_kernel=1)		
			
	def _getKeypoints(self):

		return self.detector.detect(self.img,None)
		
	def _getDescriptors(self):
		
		return self.desc.compute(self.img,self.keypoints)[1]
		
	def computeKP(self,params):
		
		self.detector,self.descriptor = self._getDetector(params)
		self.keypoints = self._getKeypoints()
		self.descriptors = self._getDescriptors()
		
		return True

		
	def setK(self,K):
		
		if K.shape == (3,3):
			
			self.K = K
		
			return True
			
		else:
			
			return 'Incorrect dimensions for K'
		
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
		
	def showImage(self):
		
		cv2.imshow('Image',self.img)
		cv2.waitKey()
		cv2.destroyAllWindows()
