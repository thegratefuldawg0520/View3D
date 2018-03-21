import image as im
import matching as mt
import os
import numpy as np
import matplotlib.pyplot as plt

def outputMatrix(mat,filename):
	
	outfile = open(filename,'w')
	
	for line in mat:

		for i in line[:-1]:
			
			outfile.write(str(i) + '\t')
			
		outfile.write(str(line[-1]) + '\n')
	outfile.close()
	
	return True

def outputKP(kpMap,filename):
	
	outfile = open(filename,'w')
	
	for keez in kpMap.keys():
		
		outfile.write(str(keez) + ':\t\t')
		outfile.write(str(kpMap[keez]) + '\n')
		
#Define the directory with the freidburg images that have been selected for this experiment
imgDir = '/home/doopy/Documents/View3D/View3D_0_1/exp1/desk'

#Pull up a list of the contents of the directory
dirList = os.listdir(imgDir)

#If folders for the keypoints and matches have not been created, create them
if not os.path.exists(imgDir + '/keypoints'):
	os.makedirs(imgDir + '/keypoints')

if not os.path.exists(imgDir + '/matches'):
	os.makedirs(imgDir + '/matches')

#Define parameters for the keypoint detection and descriptor generation
nOctaveLayers = 3
contrastThreshold = 0.04
edgeThreshold = 10
sigma = 1.6

#Define parameters for the image container object
params = {'scale':1.0,
		  'kp':'sift',
		  'nOctaveLayers':nOctaveLayers,
		  'contrastThreshold':contrastThreshold,
		  'edgeThreshold':edgeThreshold,
		  'sigma':sigma
		 }

#Currently the matrix of intrinsic parameters is set to the identity matrix, and
#keypoint detection and feature matching are performed on a distorted image.
#This is another input parameter for the image container object
K = np.eye(3)

#Define a list to store each image after loading it into an image container
#object
imgList = []
matrixList = []
#queryList = []

#For each element of the directory list, check whether it is an image. If it is
#load it into an image container object. In this case all images are .png.
for img in dirList:
	
	if img[-4:] == '.png':
		
		#Each element consists of a 2 element array [image local path and name, 
		#image object]
		imgList.append([img,im.image(imgDir + '/' + img,K,params)])

print imgList
#For each query image
for j in range(len(imgList)):
	
	#Create a column (row in data, which can be transposed after being
	#constructed) of the match matrix. The index of each row of the match matrix
	#corresponds to the index of the keypoint in the query image keypoint list,
	#the index of each column corresponds to the location of the image in the 
	#initial image list, and the value of each entry corresponds to the index of
	#the given keypoint in the given training image.
	queryImg = imgList[j]
	matchMatrix = np.zeros((len(queryImg[1].keypoints),len(imgList)))
	#For each remaining image in the image list
	for i in range(len(imgList)):

		if not i == j:
			#Match the query image to the training image
			tempMatch = mt.matches(queryImg[1],imgList[i][1],K,params)
			
			#For each match 
			for match in tempMatch.matches:
				
				#Assign the value of the location of the keypoint in the training 
				#image keypoint list to the match matrix row corresponding to the
				#index of the keypoint in the query image keypoint list
				matchMatrix[match[0].queryIdx][i] = match[0].trainIdx + 1
	
	matrixList.append(matchMatrix)
	
print '*******'
#We need to check if the point has been matched in both images, as there are
#often one to many matches in one image, and a one-to-one match in the other.
for i in range(len(matrixList)):
	
	refMatrix = matrixList[i]
	
	for j in range(refMatrix.shape[1]):
		
		for k in range(refMatrix.shape[0]):
			
			#If the match pair is not consistent in both images, set the kp
			#index in both match matrices to 0.0
			if not (matrixList[j][int(refMatrix[k][j])-1][i]-1) == k:
				
				matrixList[i][k][j] = 0.0
				
				matrixList[j][int(refMatrix[k][j])-1][i] = 0.0

keypointMap = {}
kpIndex = 0

#Now we want to associate a unique numerical index to reference each keypoint. 
#We do so by constructing a map, with integer values as the keys which reference 
#each keypoint, and the list of point coordinates and image names, for each 
#image the keypoint appears in

#Ref Image Index
for i in range(len(matrixList)):
	
	#Create a new column to store the keypoint index and set the value to -1.0
	temp = np.zeros((matrixList[i].shape[0],1))
	temp[:] = -1.0
	
	#Append it to the last column of the matrix
	matrixList[i] = np.hstack((matrixList[i],temp))

for i in range(len(matrixList)):
	#Keypoint index
	for j in range(matrixList[i].shape[0]):
		
		#Matrix Image index. -1 because we want to avoid the last column
		for k in range(i,matrixList[i].shape[1]-1):
			
			#We have a new keypoint when the value in the matchMatrix is greater
			#than 0 and all values before it in that row sum to 0 (all positive
			#integer values)
			if np.sum(matrixList[i][j][0:k]) == 0.0 and matrixList[i][j][k] > 0.0:
				
				keypointMap[kpIndex] = {}
				keypointMap[kpIndex][imgList[i][0]] = imgList[i][1].keypoints[j].pt
				keypointMap[kpIndex][imgList[k][0]] = imgList[k][1].keypoints[int(matrixList[i][j][k]) -1].pt
				matrixList[i][j][-1] = kpIndex
				
				kpIndex+=1
			
			#We have an existing keypoint when the value in the matchMatrix is 
			#greater than 0 and all values before it in that row sum to 0 (all 
			#positive integer values)
			elif matrixList[i][j][k] > 0.0 and np.sum(matrixList[i][j][0:k]) > 0.0:
				
				#Do we have the keypoint index stored in this matrix yet?
				if matrixList[i][j][-1] > -1.0:
					
					keypointMap[matrixList[i][j][-1]][imgList[k][0]] = imgList[i][1].keypoints[j].pt
					
				else:
					
					imgID = k
					kpID = -1.0
					
					#Find the first instance of the keypoint
					for l in range(len(matrixList[i][j]) - 1):
						
						#Retrieve the keypoint ID from the prior image matrix
						if matrixList[i][j][l] > 0.0:
							
							kpID = matrixList[l][int(matrixList[i][j][l])-1][-1]
							matrixList[i][j][-1] = kpID
							break
					
					for l in range(len(matrixList[i][j]) - 1):
						
						if matrixList[i][j][l] > 0.0:
							
							try:
								keypointMap[kpID][imgList[l][0]] = imgList[i][1].keypoints[j].pt
								
							except KeyError:
								print 'bamn'
								print kpID
								print i
								print j
								print k
								print l
								print matrixList[i][j][-1]
								outputMatrix(matrixList[i],imgList[i][0] + '_match.txt')
								input()
								
							except IndexError:
								print i
								print j
								print k
								print l
								print imgList[l][1]
								print int(matrixList[i][j][l]) -1
	#The last column is a column of zeros that will be used to store the global
	#keypoint index
	outputMatrix(matrixList[i],imgList[i][0] + '_match.txt')
	
outputKP(keypointMap,'kpmap.txt')
