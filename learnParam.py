import tensorflow as tf
import transforms as tn
import numpy as np

#http://patwie.com/tutorials/tensorflow-optimization.html

#Create a tensorflow session
sess = tf.Session()

#Define the model variables for sift
#nfeatures = tf.Variable([0], dtype=tf.float32)
nOctaveLayers = tf.Variable([3], dtype=tf.float32, name='oct')
contrastThreshold = tf.Variable([0.04], dtype=tf.float32, name='cont')
edgeThreshold = tf.Variable([10], dtype=tf.float32, name='edge')
sigma = tf.Variable([1.6], dtype=tf.float32,name='sigma')

x = tf.placeholder(tf.float32)

#Construct the linear model
linear_model = (nOctaveLayers + contrastThreshold + edgeThreshold + sigma)*x

#Initialize all variables to default values
init = tf.global_variables_initializer()

#Initialize the computation graph
sess.run(init)

#Create a placeholder for the variable y
y = tf.placeholder(tf.float32)
fn = (nOctaveLayers + contrastThreshold + edgeThreshold + sigma)*y

#Define the minimum sum of squares loss function
squared_deltas = tf.square(linear_model - fn)
loss = tf.reduce_sum(squared_deltas)

#Define the gradient descent optimizer and specify it to minimize the loss 
#function
optimizer = tf.train.AdamOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(tf.initialize_all_variables())

for i in range(10):
  
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

  params = {'kp':'sift','scale':0.15,'octave':3,'cont':0.04,'edge':10,'sig':1.6}
  a = tn.fundamental('/home/doopy/Documents/View3D/View3D_0_1/DJI02142.JPG','/home/doopy/Documents/View3D/View3D_0_1/DJI02141.JPG',params)
  x_ex = np.ones(len(a.bitmask))
  x_obs = np.array(a.bitmask)
  
  sess.run(train, {x: x_ex, y: x_obs})
  
print(sess.run([nOctaveLayers,contrastThreshold,edgeThreshold,sigma]))

