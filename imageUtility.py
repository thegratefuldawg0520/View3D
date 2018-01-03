import numpy as np

def toHomogeneous(pts):
	
	temp = np.ones((pts.shape[0],1,3))
	temp[:,:,:2] = pts
	
	return temp
