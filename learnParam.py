import matching as mt
import image as im
import time
import transforms as tn
import tensorflow as tf
import numpy as np

if __name__ == '__main__':

	#Create a tensorflow session
	sess = tf.Session()
	
	#Define the model variables for sift
	#*************NORMALIZE THE VARIABLES***********************************
	nOctaveLayers = tf.Variable([3], dtype=tf.float32, name='oct')
	contrastThreshold = tf.Variable([0.04], dtype=tf.float32, name='cont')
	edgeThreshold = tf.Variable([10], dtype=tf.float32, name='edge')
	sigma = tf.Variable([1.6], dtype=tf.float32,name='sigma')
	
	#Initialize x
	x = tf.placeholder(tf.float32)
	
	#Construct the linear model
	#*********************Replace this with a better model
	linear_model = (nOctaveLayers + contrastThreshold + edgeThreshold + sigma)*x
	
	#Initialize all variables to default values
	init = tf.global_variables_initializer()
	
	#Initialize the computation graph
	sess.run(init)
	
	#Create a placeholder for the variable y
	#*******************Replace this with a better model
	y = tf.placeholder(tf.float32)
	fn = (nOctaveLayers + contrastThreshold + edgeThreshold + sigma)*y
	
	#Define the loss function
	squared_deltas = tf.square(linear_model - fn)
	loss = tf.reduce_sum(squared_deltas)
	
	#Define the gradient descent optimizer and specify it to minimize the loss 
	#function
	optimizer = tf.train.AdamOptimizer(0.01)
	train = optimizer.minimize(loss)
	sess.run(tf.initialize_all_variables())

	params = {'scale':0.15, 'kp':'sift'}

	imgList = []
	
	t0 = time.time()
	
	print 'loading images'
	
	for i in range(1765,1778):
		
		tin0 = time.time()
		imgList.append(im.image('/home/dennis/Documents/View3D/images/DJI0'+ str(i) + '.JPG',params))
		tin1 = time.time()
		
		print 'image load time: ' + str(tin1 - tin0)
		
	t1 = time.time()
	
	print 'image load time: ' + str(t1 - t0)
	
	img2 = imgList[0]
	
	fundamentals = []
	costSum = 0
	
	print 'computing transformations'
	
	for i in range(10):
		
		tout0 = time.time()
		
		octave = tf.get_default_graph().get_tensor_by_name('oct:'+str(0))
		cont = tf.get_default_graph().get_tensor_by_name('cont:'+str(0))
		edge = tf.get_default_graph().get_tensor_by_name('edge:'+str(0))
		sigma = tf.get_default_graph().get_tensor_by_name('sigma:'+str(0))

		print i
		print 'octave = ' + str(sess.run(octave))
		print 'cont = ' + str(sess.run(cont))
		print 'edge = ' + str(sess.run(edge))
		print 'sigma = ' + str(sess.run(sigma))
		print '\n'
			
		for img in imgList[1:]:
			
			tin0 = time.time()
			
			img1 = img

			fundamentals.append(tn.fundamental(img1,img2,params))
			
			matchCount = fundamentals[-1].matchCount()
			inlierCount = fundamentals[-1].inlierCount()
			
			costSum = costSum + (1 - 1.0*inlierCount/matchCount)*(1 - 1.0*matchCount/10000)
			
			img2 = img1
			
			tin1 = time.time()
			
			print 'transformation time: ' + str(tin1 - tin0)
			
		tout1 = time.time()
		
		sess.run(train, {x: inlierCount, y: matchCount})
		print inlierCount
		print matchCount
		print 'bamn'
		print 'total time: ' + str(tout1 - tout0)
		
		print costSum
		
		costSum = costSum/len(fundamentals)
		
		print costSum
		
#Update the parameters and recompute the key points and descriptors
