import numpy as np
import image as im
import matching as mt
import imageUtility as ut
import transforms as tn
import matplotlib.pyplot as plt

def normalize(pts,nrow,ncol):
	
	#row
	n1 = 2.0*pts[:,0]/ncol - 1
	n2 = 2.0*pts[:,1]/nrow - 1
	
	return np.array([n1,n2])
	
def computeCollinearity(f,xo,yo,ohm,phi,kap,X,Y,Z,XP,YP,ZP,x,y):
	
	ohmR = np.radians(ohm)
	phiR = np.radians(phi)
	kapR = np.radians(kap)
	
	m11 = np.cos(phiR)*np.cos(kapR)
	m12 = np.sin(ohmR)*np.sin(phiR)*np.cos(kapR) + np.cos(ohmR)*np.sin(kapR)
	m13 = -np.cos(ohmR)*np.sin(phiR)*np.cos(kapR) + np.sin(ohmR)*np.sin(kapR)
	m21 = -np.cos(phiR)*np.sin(kapR)
	m22 = -np.sin(ohmR)*np.sin(phiR)*np.sin(kapR) + np.cos(ohmR)*np.cos(kapR)
	m23 = np.cos(ohmR)*np.sin(phiR)*np.sin(kapR) + np.sin(ohmR)*np.cos(kapR)
	m31 = np.sin(phiR)
	m32 = -np.sin(ohmR)*np.cos(phiR)
	m33 = np.cos(ohmR)*np.cos(phiR)
	
	r = m11*(XP-X) + m12*(YP-Y) + m13*(ZP-Z)
	s = m21*(XP-X) + m22*(YP-Y) + m23*(ZP-Z)
	q = m31*(XP-X) + m32*(YP-Y) + m33*(ZP-Z)
	
	print q
	print r
	print s
	
	F = xo - f*r/q
	G = yo - f*s/q

	return F,G
	
params = {'scale':0.15,
		  'kp':'sift',
		  'nOctaveLayers':3,
		  'contrastThreshold':0.04,
		  'edgeThreshold':10,
		  'sigma':1.6
		 }
		
img1 = im.image('/home/doopy/Documents/View3D/View3D_0_1/Glacier/img/EP-00-00019_0044_0002.JPG',params) 
img2 = im.image('/home/doopy/Documents/View3D/View3D_0_1/Glacier/img/EP-00-00019_0044_0003.JPG',params)

matches = mt.matches(img1,img2,params)

img1Matches = normalize(matches.matchPoints['img1'][:,0],img1.img.shape[0],img1.img.shape[1])
img2Matches = normalize(matches.matchPoints['img2'][:,0],img2.img.shape[0],img2.img.shape[1])

F,G = computeCollinearity(0.01,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.01,img1Matches[0],img1Matches[1],0,img1Matches[0],img1Matches[1])

print F
print G
