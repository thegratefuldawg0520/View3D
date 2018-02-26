import cv2
import matching as mt
import image as im
import imageUtility as ut
import time
import numpy as np
import matplotlib.pyplot as plt

ply_header = 'x y z r g b\n'

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f)

if __name__=="__main__":
	
	nOctaveLayers = 3
	dOctaveLayers = 1
	
	contrastThreshold = 0.04
	dContrastThreshold = 0.01
	
	edgeThreshold = 10
	dEdgeThreshold = 1
	
	sigma = 1.6
	dSigma = 0.1
	
	params = {'scale':0.1,
			  'kp':'sift',
			  'nOctaveLayers':nOctaveLayers,
			  'contrastThreshold':contrastThreshold,
			  'edgeThreshold':edgeThreshold,
			  'sigma':sigma
			 }
			 
	K = np.array([[1520.4, 0., 302.32], [0, 1525.9, 246.87], [0, 0, 1]])
	img1 = im.image('/home/doopy/Documents/View3D/View3D_0_1/Glacier/img/EP-00-00019_0044_0002.JPG',params)
	img1.setK(K)
	window_size = 3
	min_disp = 0
	num_disp = 112-min_disp
	
	for i in range(2,6):
		
		print i
		print i+4
		img2 = im.image('/home/doopy/Documents/View3D/View3D_0_1/Glacier/img/EP-00-00019_0044_000' + str(i + 1) + '.JPG',params)
		img2.setK(K)

		stereo = cv2.StereoSGBM_create(minDisparity=min_disp,numDisparities=num_disp,blockSize=window_size)
		
		disparity = stereo.compute(img1.img, img2.img).astype(np.float32)/16.0
		
		f = 1520.0
		Q = np.float32([[1, 0, 0, -302.32],
						[0,-1, 0,  246.87], # turn points 180 deg around x-axis,
						[0, 0, 0,     -f], # so that y-axis looks up
						[0, 0, 1,      0]])
		
		points = cv2.reprojectImageTo3D(disparity, Q)
		
		colors = cv2.cvtColor(img1.img, cv2.COLOR_BGR2RGB)
		mask = disparity > disparity.min()
		out_points = points[mask]
		out_colors = colors[mask]
		
		pts = []
		
		outfile = open('pts' + str(i) + '.txt','w')
		outfile.write('x y z r g b')
		
		for i in range(len(out_points)):
			
			if out_points[i][0] != np.inf and out_points[i][0] != -np.inf:

				outfile.write(str(out_points[i][0]) + ' ' + str(out_points[i][1]) + ' ' + str(out_points[i][2]) + ' ' + str(out_colors[i][0]) + ' ' + str(out_colors[i][1]) + ' ' + str(out_colors[i][2]) + '\n')
				
		img1 = img2
