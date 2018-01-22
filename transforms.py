import cv2
import matching as mt
import image as im
import imageUtility as ut

class transformation(object):
	
	def __init__(self, img1, img2, params, matches=None):
	
		self.img1 = self._getKeypoints(img1,params)
		self.img2 = self._getKeypoints(img2,params)
		self.matches = self._getMatches(matches,self.img1,self.img2,params)

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
		
if __name__ == '__main__':

	descList = ['kaze','sift','surf','brisk']
	temp = ['orb','daisy','kaze','freak','lucid']
		
	params = {'scale':0.15}
	
	fundamentals = []
	
	for desc in descList:
		
		params['kp'] = desc
		print desc
		fundamentals.append(fundamental('/home/doopy/Documents/View3D/View3D_0_1/0214.JPG','/home/doopy/Documents/View3D/View3D_0_1/0215.JPG',params))
		print fundamentals[-1].fundamental[1]
		fundamentals[-1].matches.drawMatches()
		
	#kp1H = ut.toHomogeneous(x.matches.matchPoints['img1'])
	#kp2H = ut.toHomogeneous(x.matches.matchPoints['img2'])
	
	
