import numpy as np

def toHomogeneous(pts):
	
	temp = np.ones((pts.shape[0],1,3))
	temp[:,:,:2] = pts
	
	return temp
	
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

