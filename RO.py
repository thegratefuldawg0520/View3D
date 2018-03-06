import numpy as np
import image as im
import matching as mt
import imageUtility as ut
import transforms as tn
import matplotlib.pyplot as plt
import cv2

def normalize(pts,nrow,ncol):
	
	#row
	n1 = 2.0*pts[:,0]/ncol - 1
	n2 = 2.0*pts[:,1]/nrow - 1
	
	return np.array([n1,n2])
	
def computeCollinearity(f,x,y,ohm,phi,kap,X,Y,Z,XP,YP,ZP):
	
	o = np.radians(ohm)
	p = np.radians(phi)
	k = np.radians(kap)
	
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
	
	Nx = r11*(XP-X) + r21*(YP-Y) + r31*(ZP-Z)
	Ny = r12*(XP-X) + r22*(YP-Y) + r32*(ZP-Z)
	D = r13*(XP-X) + r23*(YP-Y) + r33*(ZP-Z)

	F = x-f*Nx/D
	G = y-f*Ny/D
	
	return F,G
	
def uv2xy(pts,nrow,ncol):
    
    n1 = pts[:,0] - ncol/2.0
    n2 = pts[:,1] - nrow/2.0
    
    return np.array([n1,n2])
	
params = {'scale':0.15,
		  'kp':'sift',
		  'nOctaveLayers':3,
		  'contrastThreshold':0.04,
		  'edgeThreshold':10,
		  'sigma':1.6
		 }

img1 = im.image('/home/doopy/Documents/View3D/View3D_0_1/im0.png',params) 
img2 = im.image('/home/doopy/Documents/View3D/View3D_0_1/im1.png',params)

matches = mt.matches(img1,img2,params)

img1Matches = uv2xy(matches.matchPoints['img1'][:,0],img1.img.shape[0],img1.img.shape[1])
img2Matches = uv2xy(matches.matchPoints['img2'][:,0],img2.img.shape[0],img2.img.shape[1])


cx = 1424.085*params['scale'] - img1.img.shape[1]/2.0
cy = 953.053*params['scale'] - img1.img.shape[0]/2.0
f = 2852.758*params['scale']

Xl = 0
Yl = 0
Zl = f
ol = 0
pl = 0
kl = 0
Xr = 1
Yr = 0
Zr = f
Or = 0
pr = 0
kr = 0

x = [Xl,Yl,Zl,ol,pl,kl,Xr,Yr,Zr,Or,pr,kr]

for i in range(img1Matches.shape[1]):
	
	x.append(img1Matches[0,i])
	x.append(img1Matches[1,i])
	x.append(0)
	
x = np.array(x)

for j in range(0,1):
	
	A = np.zeros((4*img1Matches.shape[1],3*img1Matches.shape[1]+12))
	l = []
	
	for i in range(img1Matches.shape[1]):
		
		dXP = 0.005
		dYP = 0.005
		dZP = 0.005
	
		if not x[14+3*i] == 0:
			
			print 'bamn'
			
		#f,x,y,ohm,phi,kap,X,Y,Z,XP,YP,ZP
		fr,gr = computeCollinearity(f,cx,cy,ol,pl,kl,Xl,Yl,Zl,x[12+3*i]+dXP,x[13+3*i],x[14+3*i]) 
		fl,gl = computeCollinearity(f,cx,cy,ol,pl,kl,Xl,Yl,Zl,x[12+3*i]-dXP,x[13+3*i],x[14+3*i])
	
		A[2*i,3*i+12] = (fr - fl)/dXP
		A[2*i+1,3*i+12] = (gr - gl)/dXP
		
		#f,x,y,ohm,phi,kap,X,Y,Z,XP,YP,ZP
		fr,gr = computeCollinearity(f,cx,cy,ol,pl,kl,Xl,Yl,Zl,x[12+3*i],x[13+3*i]+dYP,x[14+3*i]) 
		fl,gl = computeCollinearity(f,cx,cy,ol,pl,kl,Xl,Yl,Zl,x[12+3*i],x[13+3*i]-dYP,x[14+3*i])
		
		A[2*i,3*i+12] = (fr - fl)/dYP
		A[2*i+1,3*i+12] = (gr - gl)/dYP
		
		#f,x,y,ohm,phi,kap,X,Y,Z,XP,YP,ZP
		fr,gr = computeCollinearity(f,cx,cy,ol,pl,kl,Xl,Yl,Zl,x[12+3*i],x[13+3*i],x[14+3*i]+dZP) 
		fl,gl = computeCollinearity(f,cx,cy,ol,pl,kl,Xl,Yl,Zl,x[12+3*i],x[13+3*i],x[14+3*i]-dZP)
		
		A[2*i,3*i+12] = (fr - fl)/dZP
		A[2*i+1,3*i+12] = (gr - gl)/dZP
		
		ptCalc = computeCollinearity(f,cx,cy,ol,pl,kl,Xl,Yl,Zl,x[12+3*i],x[13+3*i],x[14+3*i])
		l.append(img1Matches[0,i] - ptCalc[0])
		l.append(img1Matches[1,i] - ptCalc[1])
		
	for i in range(img1Matches.shape[1]):
		
		dYr = 0.005
		dZr = 0.005
		dOr = 0.5
		dpr = 0.5
		dkr = 0.5
		
		dXP = 0.005
		dYP = 0.005
		dZP = 0.005

		#f,x,y,ohm,phi,kap,X,Y,Z,XP,YP,ZP
		fr,gr = computeCollinearity(f,cx,cy,Or+dOr,pr,kr,Xr,Yr,Zr,x[12+3*i],x[13+3*i],x[14+3*i]) 
		fl,gl = computeCollinearity(f,cx,cy,Or-dOr,pr,kr,Xr,Yr,Zr,x[12+3*i],x[13+3*i],x[14+3*i])
	
		A[2*i+2*img1Matches.shape[1],9] = (fr - fl)/dOr
		A[2*i+1+2*img1Matches.shape[1],9] = (gr - gl)/dOr
		
		#f,x,y,ohm,phi,kap,X,Y,Z,XP,YP,ZP
		fr,gr = computeCollinearity(f,cx,cy,Or,pr+dpr,kr,Xr,Yr,Zr,x[12+3*i],x[13+3*i],x[14+3*i]) 
		fl,gl = computeCollinearity(f,cx,cy,Or,pr-dpr,kr,Xr,Yr,Zr,x[12+3*i],x[13+3*i],x[14+3*i])
		
		A[2*i+2*img1Matches.shape[1],10] = (fr - fl)/dpr
		A[2*i+1+2*img1Matches.shape[1],10] = (gr - gl)/dpr
		
		#f,x,y,ohm,phi,kap,X,Y,Z,XP,YP,ZP
		fr,gr = computeCollinearity(f,cx,cy,Or,pr,kr+dkr,Xr,Yr,Zr,x[12+3*i],x[13+3*i],x[14+3*i]) 
		fl,gl = computeCollinearity(f,cx,cy,Or,pr,kr+dkr,Xr,Yr,Zr,x[12+3*i],x[13+3*i],x[14+3*i])
		
		A[2*i+2*img1Matches.shape[1],11] = (fr - fl)/dkr
		A[2*i+1+2*img1Matches.shape[1],11] = (gr - gl)/dkr
		
		#f,x,y,ohm,phi,kap,X,Y,Z,XP,YP,ZP
		fr,gr = computeCollinearity(f,cx,cy,Or,pr,kr,Xr,Yr+dYr,Zr,x[12+3*i],x[13+3*i],x[14+3*i]) 
		fl,gl = computeCollinearity(f,cx,cy,Or,pr,kr,Xr,Yr-dYr,Zr,x[12+3*i],x[13+3*i],x[14+3*i])
		
		A[2*i+2*img1Matches.shape[1],7] = (fr - fl)/dYr
		A[2*i+1+2*img1Matches.shape[1],7] = (gr - gl)/dYr
		
		#f,x,y,ohm,phi,kap,X,Y,Z,XP,YP,ZP
		fr,gr = computeCollinearity(f,cx,cy,Or,pr,kr,Xr,Yr,Zr+dZr,x[12+3*i],x[13+3*i],x[14+3*i]) 
		fl,gl = computeCollinearity(f,cx,cy,Or,pr,kr,Xr,Yr,Zr+dZr,x[12+3*i],x[13+3*i],x[14+3*i])
		
		A[2*i+2*img1Matches.shape[1],8] = (fr - fl)/dZr
		A[2*i+1+2*img1Matches.shape[1],8] = (gr - gl)/dZr
		
		#f,x,y,ohm,phi,kap,X,Y,Z,XP,YP,ZP
		fr,gr = computeCollinearity(f,cx,cy,Or,pr,kr,Xr,Yr,Zr,x[12+3*i]+dXP,x[13+3*i],x[14+3*i]) 
		fl,gl = computeCollinearity(f,cx,cy,Or,pr,kr,Xr,Yr,Zr,x[12+3*i]-dXP,x[13+3*i],x[14+3*i])
		
		A[2*i+2*img1Matches.shape[1],3*i+12] = (fr - fl)/dXP
		A[2*i+1+2*img1Matches.shape[1],3*i+12] = (gr - gl)/dXP
			
		#f,x,y,ohm,phi,kap,X,Y,Z,XP,YP,ZP
		fr,gr = computeCollinearity(f,cx,cy,Or,pr,kr,Xr,Yr,Zr,x[12+3*i],x[13+3*i]+dYP,x[14+3*i]) 
		fl,gl = computeCollinearity(f,cx,cy,Or,pr,kr,Xr,Yr,Zr,x[12+3*i],x[13+3*i]-dYP,x[14+3*i])
		
		A[2*i+2*img1Matches.shape[1],3*i+13] = (fr - fl)/dYP
		A[2*i+1+2*img1Matches.shape[1],3*i+13] = (gr - gl)/dYP
			
		#f,x,y,ohm,phi,kap,X,Y,Z,XP,YP,ZP
		fr,gr = computeCollinearity(f,cx,cy,Or,pr,kr,Xr,Yr,Zr,x[12+3*i],x[13+3*i],x[14+3*i]+dZP) 
		fl,gl = computeCollinearity(f,cx,cy,Or,pr,kr,Xr,Yr,Zr,x[12+3*i],x[13+3*i],x[14+3*i]-dZP)
		
		A[2*i+2*img1Matches.shape[1],3*i+14] = (fr - fl)/dZP
		A[2*i+1+2*img1Matches.shape[1],3*i+14] = (gr - gl)/dZP
	
		ptCalc = computeCollinearity(f,cx,cy,ol,pl,kl,Xl,Yl,Zl,x[12+3*i],x[13+3*i],x[14+3*i])
		l.append(img2Matches[0,i] - ptCalc[0])
		l.append(img2Matches[1,i] - ptCalc[1])
	
	l = np.array(l)
	N = np.dot(A.T,A)
	
	print 'svd'
	Nsvd = np.linalg.svd(N)
	#print 'inv'
	Ninv = np.dot(Nsvd[2].T,np.dot(np.diag(1.0/Nsvd[1]),Nsvd[0].T))
	#print Ninv.shape
	#print 'bamn'
	#plt.imshow(A)
	#plt.show()
	#plt.imshow(N)
	#plt.show()
	#plt.imshow(Ninv)
	#plt.show()
	dx = np.dot(Ninv,np.dot(A.T,-l))
	x = x + dx

outfile = open('lspts.txt','w')
outfile.write('x y z\n')

for i in range(x[12:].shape[0]):
	
	if (i+1)%3 == 0:
		
		outfile.write(str(x[12+i]))
		outfile.write('\n')
		
	elif (i+1)%2 == 0:
		
		outfile.write(str(x[12+i]) + ' ')
		
	elif (i+1)%1 == 0:
		
		outfile.write(str(x[12+i]) + ' ')
	
	

