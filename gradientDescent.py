import numpy as np
import image as im
import matching as mt
import imageUtility as ut
import transforms as tn
import matplotlib.pyplot as plt

def openImages(imgRange,imgDir,imgParams,imgSet):
    
    images = []


    for i in imgRange:
        
        images.append(im.image(imgDir[imgSet]['dir'] + str(i) + imgDir[imgSet]['ext'],imgParams))
        
    return images

def computeOutliers(imageSet,imgParams):

	print 'Computing Outliers'
	print imgParams
	print '\n'
	
	for img in imageSet:

		img.computeKP(imgParams)

	img1 = imageSet[0]
	inlierSum = 0
	outlierSum = 0

	for img2 in imageSet[1:]:

		F = tn.fundamental(img1,img2,imgParams)

		inlierSum += F.inlierCount()
		outlierSum += F.outlierCount()

		img1 = img2   

	return outlierSum
	
nOctaveLayers = 3
contrastThreshold = 0.04
edgeThreshold = 10
sigma = 1.6

imgDir = {'glacier':{'dir':'/home/doopy/Documents/View3D/View3D_0_1/exp1/glacier/','ext':'.JPG'},
            'wbnp':{'dir':'/home/doopy/Documents/View3D/View3D_0_1/exp1/wbnp/','ext':'.JPG'},
            'desk':{'dir':'/home/doopy/Documents/View3D/View3D_0_1/exp1/desk/','ext':'.png'}
           }

imgRange = range(1,10)

imgParams = {'scale':0.15,
          'kp':'sift',
          'nOctaveLayers':3,
          'contrastThreshold':0.04,
          'edgeThreshold':10,
          'sigma':1.6
         }

params = {'nOctaveLayers':{'step':1,'bdy':6},
        'contrastThreshold':{'step':0.01,'bdy':0.09},
        'edgeThreshold':{'step':1,'bdy':16},
        'sigma':{'step':0.1,'bdy':2.1}
         }

imageSet = openImages(imgRange,imgDir,imgParams,'glacier')

minima = False

while not minima:

	print 'Computing y'
	y = computeOutliers(imageSet,imgParams)
	print y
	print '\n'
	
	gradient = {}
	
	for parameter in params.keys():
		
		print parameter

		gradParams = {key: value for key, value in imgParams.items()}
		
		gradient[parameter] = {}
		
		gradParams[parameter] = imgParams[parameter] - params[parameter]['step']
		
		if gradParams[parameter] > 0:

			a = computeOutliers(imageSet,gradParams)
			print a
			gradient[parameter]['left'] = (a - y)			

		else:
			
			gradient[parameter]['left'] = 0

		gradParams[parameter] = imgParams[parameter] + params[parameter]['step']

		if gradParams[parameter] < params[parameter]['bdy']:
			
			a = computeOutliers(imageSet,gradParams)
			print a
			gradient[parameter]['right'] = (a - y)
    		
		else:
    		
			gradient[parameter]['right'] = 0
			
	print 'gradient'
	print gradient
	input()
	
	minima = True
	
	for parameter in gradient.keys():
    	
		print parameter + ' gradient'
		rgrad = gradient[parameter]['right']
		print rgrad
		lgrad = gradient[parameter]['left']
		print lgrad
    	
		if lgrad < rgrad and lgrad < 0:
    		
			print 'rgrad'
			imgParams[parameter] = imgParams[parameter] - params[parameter]['step']
			minima = False
    	
		elif rgrad < lgrad and rgrad < 0:
    		
			print 'lgrad'
			imgParams[parameter] = imgParams[parameter] + params[parameter]['step']
			minima = False
    		
	print 'end of loop'

	input()
    	
    	

