import numpy as np

def toHomogeneous(pts):
	
	temp = np.ones((pts.shape[0],3))
	temp[:,0] = pts.T[0]
	temp[:,1] = pts.T[1]
	
	return temp
	
def normalize(pts,nrow,ncol):
	
	#row
	n1 = 2.0*pts[:,0]/ncol - 1
	n2 = 2.0*pts[:,1]/nrow - 1
	
	return np.array([n1,n2])

def eulerAngles(R):
	
	p = np.arcsin(R[0,2])
	o = np.arctan2(-R[1,2],R[2,2])
	k = np.arctan2(-R[0,1],R[0,0])
	
	return np.array([o,p,k])
		
def loadCameras(filename):
	
	ext = filename.split('.')[-1]
	
	inFile = open(filename)
	
	if ext == 'txt':
		
		inFile.readline()
		fields = inFile.readline().split()
		
		imageDict = {}
		
		for line in inFile:
			
			entry = line.split()
			newImage = {}
			
			newImage['pt'] = np.array([entry[1],entry[2],entry[3]],dtype=np.float32)
			newImage['opk'] = np.array([entry[4],entry[5],entry[6]],dtype=np.float32)
			newImage['R'] = np.array([[entry[7],entry[8],entry[9]],[entry[10],entry[11],entry[12]],[entry[13],entry[14],entry[15]]],dtype=np.float32)
			
			imageDict[entry[0]] = newImage
			
		return imageDict

def loadCamerasTemple(filename):
	
	infile = open(filename)
	cameras = infile.readline()
	
	cameras = {}
	
	for line in infile:
		
		splitLine = line.split()
		
		cameras[splitLine[0]] = {}
		cameras[splitLine[0]]['K'] = np.array([splitLine[1:4],splitLine[4:7],splitLine[7:10]],dtype=np.float32)
		cameras[splitLine[0]]['R'] = np.array([splitLine[10:13],splitLine[13:16],splitLine[16:19]],dtype=np.float32)
		cameras[splitLine[0]]['t'] = np.array(splitLine[19:22],dtype=np.float32)
		
	return cameras
	
def createPCFile(pc, filename):
	
	outfile = open(filename,'w')
	
	outfile.write('X Y Z\n')
	
	for pt in pc:
		
		#print pt.dot(pt)
		#print pt
		outfile.write(str(pt[0]) + ' ' + str(pt[1]) + ' ' + str(pt[2]) + '\n')
		
	outfile.close()
	
	return True
			
#loadCameras('/home/doopy/Documents/View3D/View3D_0_1/Glacier/Processing Results/Cameras/cameras.txt')

